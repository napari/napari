from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

from pydantic import BaseModel, BaseSettings

from ...utils.events import EmitterGroup, EventedModel
from ...utils.translations import trans

if TYPE_CHECKING:
    from typing import AbstractSet, Any, Union

    from pydantic.env_settings import SettingsSourceCallable

    from ...utils.events import Event

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = Dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]

_NOT_SET = object()


class EventedSettings(BaseSettings, EventedModel):
    """A variant of EventedModel designed for settings.

    Pydantic's BaseSettings model will attempt to determine the values of any
    fields not passed as keyword arguments by reading from the environment.
    """

    # provide config_path=None to prevent reading from disk.
    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        self.events.add(changed=None)

        # re-emit subfield
        for name in self.__fields__:
            attr = getattr(self, name)
            if isinstance(getattr(attr, 'events', None), EmitterGroup):
                attr.events.connect(partial(self._on_sub_event, field=name))

    def _on_sub_event(self, event: Event, field=None):
        """emit the field.attr name and new value"""
        if field:
            field += "."
        value = getattr(event, 'value', None)
        self.events.changed(key=f'{field}{event._type}', value=value)


class EventedConfigFileSettings(EventedSettings):
    """This adds config read/write and yaml support to EventedSettings.

    If your settings class *only* needs to read variables from the environment,
    such as environment variables (but not a config file), then subclass from
    EventedSettings.
    """

    _config_path: Optional[Path] = None
    _save_on_change: bool = True

    # provide config_path=None to prevent reading from disk.
    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        _cfg = (
            config_path
            if config_path is not _NOT_SET
            else self.__private_attributes__['_config_path'].get_default()
        )
        # this line is here for usage in the `customise_sources` hook.  It
        # will be overwritten in __init__ by BaseModel._init_private_attributes
        # so we set it again after __init__.
        self._config_path = _cfg
        super().__init__(**values)
        self._config_path = _cfg

    def _on_sub_event(self, event, field=None):
        super()._on_sub_event(event, field)
        if self._save_on_change:
            self.save()

    @property
    def config_path(self):
        return self._config_path

    def dict(  # type: ignore
        self,
        *,
        include: Union[AbstractSetIntStr, MappingIntStrAny] = None,  # type: ignore
        exclude: Union[AbstractSetIntStr, MappingIntStrAny] = None,  # type: ignore
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
    ) -> DictStrAny:
        data = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        if exclude_env:
            eset = getattr(self.__config__, '_env_settings', None)
            if callable(eset):
                env_data = eset(type(self))
                if env_data:
                    _nested_del(data, env_data)
        return data

    def yaml(
        self,
        *,
        include: Union[AbstractSetIntStr, MappingIntStrAny] = None,  # type: ignore
        exclude: Union[AbstractSetIntStr, MappingIntStrAny] = None,  # type: ignore
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
        **dumps_kwargs: Any,
    ) -> str:
        import json

        import yaml

        data = self.dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_env=exclude_env,
        )
        _remove_empty_dicts(data)

        # We roundtrip to keep custom string objects (like SchemaVersion)
        # yaml representable
        # FIXME: should provide yaml serializer on field itself (as for json)
        _json = self.__config__.json_dumps(data, default=self.__json_encoder__)
        return yaml.safe_dump(json.loads(_json), **dumps_kwargs)

    def save(self, path=None, **dict_kwargs):
        path = path or self.config_path
        if not path:
            raise ValueError("No path provided in config or save argument.")

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(exist_ok=True, parents=True)

        dict_kwargs.setdefault('exclude_defaults', True)
        dict_kwargs.setdefault('exclude_env', True)
        data = self.dict(**dict_kwargs)
        _remove_empty_dicts(data)

        dump = _get_io_func_for_path(path, 'dump')
        if dump is not None:
            with open(path, 'w') as target:
                dump(data, target)

    class Config:
        sources: Sequence[str] = []
        _env_settings: SettingsSourceCallable

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            """customise the way data is loaded.

            This does 2 things:
            1) adds the `config_file_settings_source` to the sources, which
               will load data from `settings._config_path` if it exists.
            2) adds support for nested env_vars, such that if a model with an
               env_prefix of "foo_" has a field named `bar`, then you can use
               `FOO_BAR_X=1` to set the x attribute in `foo.bar`.

            Priority is given to sources earlier in the list.  You can resort
            the return list to change the priority of sources.
            """
            nested_eset = nested_env_settings(env_settings)
            cls._env_settings = nested_eset
            return (
                init_settings,
                nested_eset,
                config_file_settings_source,
                file_secret_settings,
            )


# Utility functions


def nested_env_settings(super_eset=None) -> SettingsSourceCallable:
    """Wraps the pydantic EnvSettingsSource to support nested env vars.

    for example:
    `NAPARI_APPEARANCE_THEME=light`
    will parse to:
    {'appearance': {'theme': 'light'}}

    currently only supports one level of nesting
    """

    def _inner(settings: BaseSettings) -> Dict[str, Any]:
        d = super_eset(settings)

        if settings.__config__.case_sensitive:
            env_vars: Mapping[str, Optional[str]] = os.environ
        else:
            env_vars = {k.lower(): v for k, v in os.environ.items()}

        for name, field in settings.__fields__.items():
            if not isinstance(field.type_, type(BaseModel)):
                continue
            for env_name in field.field_info.extra['env_names']:
                for sf in field.type_.__fields__:
                    env_val = env_vars.get(f'{env_name}_{sf.lower()}')
                    if env_val is not None:
                        break
                if env_val is not None:
                    if field.alias not in d:
                        d[field.alias] = {}
                    d[field.alias][sf] = env_val
        return d

    return _inner


def config_file_settings_source(settings: BaseSettings) -> dict:
    """Read config files"""
    _path = getattr(settings, '_config_path', None)
    sources = list(getattr(settings.__config__, 'sources', []))
    if _path:
        sources.append(_path)
    if not sources:
        return {}

    data: dict = {}
    for path in sources:
        _path = Path(path).expanduser().resolve()
        if not _path.is_file():
            # warnings.warn(
            #     trans._(
            #         "Requested config path is not a file: {path}",
            #         deferred=True,
            #         path=_path,
            #     )
            # )
            continue

        load = _get_io_func_for_path(_path, 'load')
        if load is None:
            continue

        try:
            new_data = load(_path.read_text()) or {}
        except Exception as err:
            warnings.warn(
                trans._(
                    "The content of the napari settings file could not be read\n\nThe default settings will be used and the content of the file will be replaced the next time settings are changed.\n\nError:\n{err}",
                    deferred=True,
                    err=err,
                )
            )
            continue
        assert isinstance(new_data, dict), _path.read_text()
        _nested_merge(data, new_data, copy=False)
    return data


def _nested_merge(dct: dict, merge_dct: dict, copy=True):
    """merge nested dict keys"""
    _dct = dct.copy() if copy else dct
    for k, v in merge_dct.items():
        if k in _dct and isinstance(dct[k], dict) and isinstance(v, dict):
            _nested_merge(_dct[k], v, copy=False)
        else:
            _dct[k] = v
    return _dct


def _nested_del(dct: dict, del_dct: dict):
    """delete nested dict keys"""
    for k, v in del_dct.items():
        if dct.get(k) == v:
            del dct[k]
        elif isinstance(v, dict):
            _nested_del(dct[k], v)
    return dct


def _remove_empty_dicts(dct: dict, recurse=True):
    """Remove all (nested) keys with empty dict values from `dct`"""
    for k, v in list(dct.items()):
        if isinstance(v, Mapping) and recurse:
            _remove_empty_dicts(dct[k])
        if v == {}:
            del dct[k]
    return dct


def _get_io_func_for_path(path: Union[str, Path], mode='dump'):
    """get json/yaml [safe_]load/[safe_]dump for `path`"""
    assert mode in {'dump', 'load'}
    if str(path).endswith(('.yaml', '.yml')):
        return getattr(__import__('yaml'), f'safe_{mode}')
    if str(path).endswith(".json"):
        return getattr(__import__('json'), mode)

    warnings.warn(
        trans._(
            "Unrecognized file extension for config_path: {path}", path=path
        )
    )
    return None

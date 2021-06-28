from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import BaseModel, BaseSettings, root_validator
from pydantic.main import ModelMetaclass

from ...utils.events import EventedModel
from ...utils.translations import trans

if TYPE_CHECKING:
    from pydantic.env_settings import SettingsSourceCallable

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = Dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]


class BaseNapariSettings(BaseSettings, EventedModel):
    class Config:
        load_from: Sequence[str] = []
        save_to: str = ''
        _env_settings: SettingsSourceCallable

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            nested_eset = nested_env_settings(env_settings)

            cls._env_settings = nested_eset
            return (
                init_settings,
                nested_eset,
                config_file_settings_source,
                file_secret_settings,
            )

    @root_validator(pre=True)
    def _fill_optional_fields(cls, values):
        # allow BaseModel subfields to be declared without default_factory
        # if they have no required fields
        for name, field in cls.__fields__.items():
            if isinstance(field.type_, type(BaseModel)) and not any(
                sf.required for sf in field.type_.__fields__.values()
            ):
                values.setdefault(name, {})
        return values

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
                    nested_del(data, env_data)
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
        remove_empty_dicts(data)

        # We roundtrip to keep custom string objects (like SchemaVersion)
        # yaml representable
        # FIXME: should provide yaml serializer on field itself (as for json)
        _json = self.__config__.json_dumps(data, default=self.__json_encoder__)
        return yaml.safe_dump(json.loads(_json), **dumps_kwargs)

    def save(self, path=None, **dict_kwargs):
        path = path or getattr(self.__config__, 'save_to', None)
        if not path:
            raise ValueError("No path provided in config or save argument.")

        dict_kwargs.setdefault('exclude_defaults', True)
        dict_kwargs.setdefault('exclude_env', True)
        data = self.dict(**dict_kwargs)
        remove_empty_dicts(data)

        dump = _get_io_func_for_path(path, 'dump')
        with open(path, 'w') as target:
            dump(data, target)

    @property
    def _defaults(self):
        return get_defaults(self)

    def reset(self):
        for name, value in self._defaults.items():
            setattr(self, name, value)


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
    data: dict = {}
    for path in getattr(settings.__config__, 'load_from', []):
        _path = Path(path)
        if not _path.is_file():
            continue

        try:
            load = _get_io_func_for_path(_path, 'load')
        except ValueError as e:
            warnings.warn(str(e))
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

        nested_merge(data, new_data, copy=False)
    return data


def nested_merge(dct: dict, merge_dct: dict, copy=True):
    _dct = dct.copy() if copy else dct
    for k, v in merge_dct.items():
        if k in _dct and isinstance(dct[k], dict) and isinstance(v, dict):
            nested_merge(_dct[k], v, copy=False)
        else:
            _dct[k] = v
    return _dct


def nested_del(dct: dict, del_dct: dict):
    for k, v in del_dct.items():
        if dct.get(k) == v:
            del dct[k]
        elif isinstance(v, dict):
            nested_del(dct[k], v)
    return dct


def _get_io_func_for_path(path: Union[str, Path], mode='dump'):
    assert mode in {'dump', 'load'}
    if str(path).endswith(('.yaml', '.yml')):
        return getattr(__import__('yaml'), f'safe_{mode}')
    if str(path).endswith(".json"):
        return getattr(__import__('json'), mode)
    raise ValueError(f"Unrecognized file extension: {path}")


def remove_empty_dicts(dct: dict, recurse=True):
    for k, v in list(dct.items()):
        if isinstance(v, Mapping) and recurse:
            remove_empty_dicts(dct[k])
        if v == {}:
            del dct[k]
    return dct


def get_defaults(obj: BaseModel):
    dflt = {}
    for k, v in obj.__fields__.items():
        d = v.get_default()
        if d is None and isinstance(v.type_, ModelMetaclass):
            d = get_defaults(v.type_)
        dflt[k] = d
    return dflt

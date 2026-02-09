from __future__ import annotations

import contextlib
import json
import logging
import os
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from warnings import warn

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
)
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from napari._pydantic_util import get_inner_type, get_origin
from napari.settings._yaml import PydanticYamlMixin
from napari.utils.compat import StrEnum
from napari.utils.events import EmitterGroup, EventedModel
from napari.utils.misc import StringEnum, deep_update
from napari.utils.translations import trans

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Union

    # TODO: needs to be fixed properly
    SettingsSourceCallable = Any

    from napari.utils.events import Event

    IntStr = Union[int, str]
    from pydantic.main import IncEx

    DictStrAny = dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]
    JSONable = str | list | dict | int | float | bool | None

Dict = dict  # rename, because EventedSettings has method dict


def _json_encode(
    dkt: dict[type, Callable[[Any], JSONable]],
) -> Callable[[Any], JSONable]:
    def json_encode(value: Any) -> JSONable:
        if type(value) in dkt:
            return dkt[type(value)](value)
        if isinstance(value, (StrEnum, StringEnum)):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return [str(x) for x in value]

        raise TypeError(
            f'Object of type {type(value)} is not JSON serializable'
        )

    return json_encode


class EventedSettings(BaseSettings, EventedModel):
    """A variant of EventedModel designed for settings.

    Pydantic's BaseSettings model will attempt to determine the values of any
    fields not passed as keyword arguments by reading from the environment.
    """

    # provide config_path=None to prevent reading from disk.

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        self.events.add(changed=None)
        self._connect(self)

    @staticmethod
    def _warn_restart(*_: Event) -> None:
        warn(
            trans._(
                'Restart required for this change to take effect.',
                deferred=True,
            )
        )

    def _connect(self, model: EventedModel, prefix: str = '') -> None:
        """Recursively connect and re-emit to all sub-fields."""
        for name, field in model.__class__.model_fields.items():
            attr = getattr(model, name)
            if isinstance(getattr(attr, 'events', None), EmitterGroup):
                path = f'{prefix}{name}'
                attr.events.connect(partial(self._on_sub_event, field=path))
                self._connect(attr, f'{path}.')

            extra = getattr(field, 'json_schema_extra', None)
            if extra is not None and extra.get('requires_restart', False):
                emitter = getattr(model.events, name)
                emitter.connect(self._warn_restart)

    def _on_sub_event(self, event: Event, field=None):
        """emit the field.attr name and new value"""
        if field:
            field += '.'
        value = getattr(event, 'value', None)
        self.events.changed(key=f'{field}{event._type}', value=value)


class _NotSetType:
    def __bool__(self) -> bool:
        return False


_NOT_SET = _NotSetType()


class FileConfigSettingsSource(PydanticBaseSettingsSource):
    """Class to load settings from a config file (yaml, json)."""

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Required to satisfy PydanticBaseSettingsSource interface.
        All logic is in __call__ method.
        """
        raise NotImplementedError

    def __call__(self) -> Dict[str, Any]:
        sources: list[str | Path] = list(
            getattr(self.settings_cls.model_config, 'sources', [])
        )
        default_cfg = self.settings_cls.model_fields['config_path'].default

        filepath = self.current_state.get('config_path')

        if filepath not in {_NOT_SET, None} and Path(filepath).exists():
            sources.append(filepath)

        if not sources:
            return {}

        data: dict = {}

        for path in sources:
            path_ = Path(path).expanduser().resolve()

            if not path_.is_file():
                # if it wasn't the `_config_path` stated in the BaseModel itself,
                # we warn, since this would have been user provided.
                if path_ != default_cfg:
                    _logger.warning(
                        trans._(
                            'Requested config path is not a file: {path}',
                            path=path_,
                        )
                    )
                continue
                # get loader for yaml/json
            if path_.suffix in {'.yaml', '.yml'}:
                load = __import__('yaml').safe_load
            elif path_.suffix == '.json':
                load = __import__('json').load
            else:
                warn(
                    trans._(
                        'Unrecognized file extension for config_path: {path}',
                        path=path,
                    )
                )
                continue

            try:
                # try to parse the config file into a dict
                new_data = load(path_.read_text()) or {}
            except Exception as err:  # noqa: BLE001
                _logger.warning(
                    trans._(
                        'The content of the napari settings file could not be read\n\nThe default settings will be used and the content of the file will be replaced the next time settings are changed.\n\nError:\n{err}',
                        deferred=True,
                        err=err,
                    )
                )
                continue
            assert isinstance(new_data, dict), path_.read_text()
            deep_update(data, new_data, copy=False)

        self.validate_settings_kwargs(data, filepath)

        data['config_file_settings'] = data.copy()
        return data

    def validate_settings_kwargs(
        self, data: dict[str, Any], path_: Path
    ) -> dict[str, Any]:
        try:
            # validate the data, passing config_path=None so we dont recurse
            # back to this point again.
            self.settings_cls(config_path=None, **data)
        except ValidationError as err:
            if self.settings_cls.model_config.get(
                'strict_config_check', False
            ):
                raise

            # if errors occur, we still want to boot, so we just remove bad keys
            errors = err.errors()
            msg = trans._(
                'Validation errors in config file(s).\nThe following fields have been reset to the default value:\n\n{errors}\n',
                deferred=True,
                errors=errors,  # TODO: is this good enough?
            )
            with contextlib.suppress(Exception):
                # we're about to nuke some settings, so just in case... try backup
                backup_path = path_.parent / f'{path_.stem}.BAK{path_.suffix}'
                backup_path.write_text(path_.read_text())

            _logger.warning(msg)
            try:
                _remove_bad_keys(data, [e.get('loc', ()) for e in errors])
            except KeyError:  # pragma: no cover
                _logger.warning(
                    trans._(
                        'Failed to remove validation errors from config file. Using defaults.'
                    )
                )
                data = {}
        return data


class NapariEnvSettingsSource(EnvSettingsSource):
    def __call__(self):
        res = super().__call__()
        env_lower = {k.lower(): v for k, v in os.environ.items()}
        self.scan_env_aliases(res, self.settings_cls, [], env_lower)

        res['env_settings'] = res.copy()
        return res

    def scan_env_aliases(
        self,
        dkt: dict[str, Any],
        class_: type[BaseSettings],
        path: list[str],
        env_dkt: dict[str, str],
    ):
        for field_name, field in class_.model_fields.items():
            if field.exclude:
                continue
            field_type = get_inner_type(field.annotation)
            if get_origin(field_type) is None and issubclass(
                field_type, BaseModel
            ):
                self.scan_env_aliases(
                    dkt, field_type, path + [field_name], env_dkt
                )
            if field.validation_alias is None:
                continue
            if (
                not isinstance(field.validation_alias, AliasChoices)
                or field.validation_alias.choices[0] != field_name
            ):
                raise ValueError(
                    f'Invalid validation alias for field {field_name} needs to be AliasChoices with first choice as field name'
                )
            for env_name in field.validation_alias.choices[1:]:
                if env_name not in env_dkt:
                    continue
                env_value = env_dkt[str(env_name)]
                value = TypeAdapter(field_type).validate_python(env_value)
                sub_dkt = dkt
                for sub in path:
                    sub_dkt = sub_dkt.setdefault(sub, {})
                if field_name in sub_dkt:
                    _logger.warning(
                        'Multiple environment variables found for %(field_name) at %(env_name) and %(existing_env_name). Using earlier value.',
                        extra={
                            'field_name': field_name,
                            'env_name': env_name,
                            'existing_env_name': field.validation_alias.choices,
                        },
                    )
                    continue
                sub_dkt[field_name] = value


class EventedConfigFileSettings(EventedSettings, PydanticYamlMixin):
    """This adds config read/write and yaml support to EventedSettings.

    If your settings class *only* needs to read variables from the environment,
    such as environment variables (but not a config file), then subclass from
    EventedSettings.
    """

    config_path: Path | _NotSetType | None = Field(default=None, exclude=True)
    env_settings: Dict = Field(
        default_factory=dict, exclude=True, repr=False, frozen=True
    )
    _save_on_change: bool = PrivateAttr(True)
    # this dict stores the data that came specifically from the config file.
    # it's populated in `config_file_settings_source` and
    # used in `_remove_env_settings`
    config_file_settings: Dict = Field(
        default_factory=dict, exclude=True, repr=False, frozen=True
    )

    # provide config_path=None to prevent reading from disk.
    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        cfg = (
            config_path
            if config_path is not _NOT_SET
            else self.__class__.model_fields['config_path'].get_default()
        )
        # this line is here for usage in the `customise_sources` hook.  It
        # will be overwritten in __init__ by BaseModel._init_private_attributes
        # so we set it again after __init__.
        # self._config_path = _cfg
        if 'env_settings' in values:
            raise ValueError('env_settings is a reserved field name')
        super().__init__(config_path=cfg, **values)

    def _maybe_save(self):
        if self._save_on_change and self.config_path:
            self.save()

    def _on_sub_event(self, event, field=None):
        super()._on_sub_event(event, field)
        self._maybe_save()

    def dict(
        self,
        *,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
    ) -> DictStrAny:
        """Return dict representation of the model.

        May optionally specify which fields to include or exclude.

        .. deprecated:: 0.7.0
              `dict` will be removed in napari 0.8.0 it is replaced by
              `model_dump` following pydantic 1 to 2 changes.
        """
        warnings.warn(
            'method `dict` is deprecated in 0.7.0 and will be removed in napari 0.8.0, use `model_dump` instead.',
            category=FutureWarning,
        )
        return self.model_dump(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_env=exclude_env,
        )

    def model_dump(
        self,
        *,
        mode: Literal['json', 'python'] | str = 'python',  # noqa: PYI051
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False,
        exclude_env: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return dict representation of the model.

        May optionally specify which fields to include or exclude.

        For `exclude_env` kwarg: if True, will exclude any settings
        provided by environment variables.

        For other kwargs, see ``pydantic.BaseModel.model_dump`` docs:
        https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump
        """
        data = super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            serialize_as_any=serialize_as_any,
            **kwargs,
        )
        if exclude_env:
            self._remove_env_settings(data)
        return data

    def _save_dict(self, **dict_kwargs: Any) -> DictStrAny:
        """The minimal dict representation that will be persisted to disk.

        By default, this will exclude settings values that match the default
        value, and will exclude values that were provided by environment
        variables.  Empty dicts will also be removed.
        """
        dict_kwargs.setdefault('exclude_defaults', True)
        dict_kwargs.setdefault('exclude_env', True)
        data = self.model_dump(**dict_kwargs)
        _remove_empty_dicts(data)
        return data

    def save(
        self, path: str | Path | _NotSetType | None = None, **dict_kwargs
    ):
        """Save current settings to path.

        By default, this will exclude settings values that match the default
        value, and will exclude values that were provided by environment
        variables.  (see `_save_dict` method.)
        """
        path = path or self.config_path
        # use insinstance so mypy is happy
        if not path or isinstance(path, _NotSetType):
            raise ValueError(
                trans._(
                    'No path provided in config or save argument.',
                    deferred=True,
                )
            )

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(exist_ok=True, parents=True)
        self._dump(str(path), self._save_dict(**dict_kwargs))

    def _dump(self, path: str, data: Dict) -> None:
        """Encode and dump `data` to `path` using a path-appropriate encoder."""
        if str(path).endswith(('.yaml', '.yml')):
            data_ = self._yaml_dump(data)
        elif str(path).endswith('.json'):
            data_ = json.dumps(
                data, default=_json_encode(self.model_config['json_encoders'])
            )
        else:
            raise NotImplementedError(
                trans._(
                    'Can only currently dump to `.json` or `.yaml`, not {path!r}',
                    deferred=True,
                    path=path,
                )
            )
        with open(path, 'w') as target:
            target.write(data_)

    def _remove_env_settings(self, data):
        """Remove key:values from `data` that match settings from env vars.

        This is handy when we want to persist settings to disk without
        including settings that were provided by environment variables (which
        are usually more temporary).
        """
        if self.env_settings:
            _restore_config_data(
                data, self.env_settings, self.config_file_settings
            )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            NapariEnvSettingsSource(settings_cls),
            FileConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    model_config = SettingsConfigDict(
        strict_config_check=False,
    )


def _remove_bad_keys(data: dict, keys: list[tuple[int | str, ...]]):
    """Remove list of keys (as string tuples) from dict (in place).

    Parameters
    ----------
    data : dict
        dict to modify (will be modified inplace)
    keys : List[Tuple[str, ...]]
        list of possibly nested keys

    Examples
    --------

    >>> data = {'a': 1, 'b' : {'c': 2, 'd': 3}, 'e': 4}
    >>> keys = [('b', 'd'), ('e',)]
    >>> _remove_bad_keys(data, keys)
    >>> data
    {'a': 1, 'b': {'c': 2}}

    """
    for key in keys:
        if not key:
            continue  # pragma: no cover
        d = data
        while True:
            base, *key = key  # type: ignore
            if not key:
                break
            # since no pydantic fields will be integers, integers usually
            # mean we're indexing into a typed list. So remove the base key
            if isinstance(key[0], int):
                break
            d = d[base]
        del d[base]


def _restore_config_data(dct: dict, delete: dict, defaults: dict) -> dict:
    """delete nested dict keys, restore from defaults."""
    for k, v in delete.items():
        # recurse
        if isinstance(v, dict):
            dflt = defaults.get(k, {})
            if not isinstance(dflt, dict):
                dflt = {}
            _restore_config_data(dct.setdefault(k, {}), v, dflt)
        # restore from defaults if present, or just delete the key
        elif k in defaults:
            dct[k] = defaults[k]
        elif k in dct:
            del dct[k]

    return dct


def _remove_empty_dicts(dct: dict, recurse=True) -> dict:
    """Remove all (nested) keys with empty dict values from `dct`"""
    for k, v in list(dct.items()):
        if isinstance(v, Mapping) and recurse:
            _remove_empty_dicts(dct[k])
        if v == {}:
            del dct[k]
    return dct

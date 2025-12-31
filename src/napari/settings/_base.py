from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from warnings import warn

from pydantic import ConfigDict

from napari._pydantic_compat import (
    PrivateAttr,
    ValidationError,
    display_errors,
)
from napari.settings._yaml import PydanticYamlMixin
from napari.utils.events import EmitterGroup, EventedModel
from napari.utils.misc import deep_update
from napari.utils.translations import trans

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from typing import Union

    from napari.utils.events import Event

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]

Dict = dict  # rename, because EventedSettings has method dict


def _exclude_defaults_evented(
    obj: EventedModel,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Remove fields from data that equal their defaults.

    Pydantic V2's exclude_defaults doesn't work correctly for EventedModel
    because it compares __pydantic_private__ (private attributes) which are
    always different between instances. This function manually compares using
    only public model fields.
    """
    from pydantic.fields import PydanticUndefined

    result = {}
    for field_name, field_info in type(obj).model_fields.items():
        if field_name not in data:
            continue
        current_value = getattr(obj, field_name)
        default_value = field_info.default

        # If default is PydanticUndefined, check for default_factory
        if default_value is PydanticUndefined:
            if field_info.default_factory is not None:
                # Create a default instance to compare against
                # default_factory is a no-arg callable in Pydantic V2
                factory = field_info.default_factory
                default_value = factory()  # type: ignore[call-arg]
            else:
                # No default available, include the field
                result[field_name] = data[field_name]
                continue

        # For nested EventedModels, use their custom __eq__ which compares only fields
        if isinstance(current_value, EventedModel):
            if current_value == default_value:
                # Skip - equals default
                continue
            # Recurse for nested models
            nested_data = _exclude_defaults_evented(
                current_value, data[field_name]
            )
            if nested_data:  # Only include if there's non-default data
                result[field_name] = nested_data
        else:
            # For non-EventedModel fields, compare values directly
            if current_value != default_value:
                result[field_name] = data[field_name]

    return result


class SettingsError(ValueError):
    """Error raised when settings validation fails."""


class EventedSettings(EventedModel):
    """A variant of EventedModel designed for settings.

    Pydantic's BaseSettings model will attempt to determine the values of any
    fields not passed as keyword arguments by reading from the environment.

    Note: In Pydantic V2, BaseSettings is in a separate package (pydantic-settings).
    We inherit from EventedModel and add settings-like behavior.
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='ignore',
    )

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
        # Use type(model).model_fields to avoid deprecation warning in V2.11+
        for name, field_info in type(model).model_fields.items():
            attr = getattr(model, name)
            if isinstance(getattr(attr, 'events', None), EmitterGroup):
                path = f'{prefix}{name}'
                attr.events.connect(partial(self._on_sub_event, field=path))
                self._connect(attr, f'{path}.')

            # Check for requires_restart in json_schema_extra
            extra = field_info.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get('requires_restart'):
                emitter = getattr(model.events, name)
                emitter.connect(self._warn_restart)

    def _on_sub_event(self, event: Event, field=None):
        """emit the field.attr name and new value"""
        if field:
            field += '.'
        value = getattr(event, 'value', None)
        self.events.changed(key=f'{field}{event._type}', value=value)


_NOT_SET = object()


class EventedConfigFileSettings(EventedSettings, PydanticYamlMixin):
    """This adds config read/write and yaml support to EventedSettings.

    If your settings class *only* needs to read variables from the environment,
    such as environment variables (but not a config file), then subclass from
    EventedSettings.
    """

    _config_path: Path | None = PrivateAttr(default=None)
    _save_on_change: bool = PrivateAttr(default=True)
    # this dict stores the data that came specifically from the config file.
    # it's populated in `config_file_settings_source` and
    # used in `_remove_env_settings`
    _config_file_settings: dict = PrivateAttr(default_factory=dict)
    # Store env settings for later removal
    _env_settings_cache: dict = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='ignore',
    )
    # Settings-specific config (not part of Pydantic V2 ConfigDict)
    # Use ClassVar to prevent Pydantic from treating this as a PrivateAttr
    _env_prefix: ClassVar[str] = 'NAPARI_'

    # provide config_path=None to prevent reading from disk.
    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        import copy as copy_module

        # Determine the config path to use
        # Use provided config_path if given, otherwise None
        _cfg = config_path if config_path is not _NOT_SET else None

        # Load settings from config file if path exists
        original_file_settings = {}
        if _cfg is not None:
            file_settings = config_file_settings_source(self.__class__, _cfg)
            # Store original file settings before merging
            original_file_settings = copy_module.deepcopy(file_settings)
            # Merge with provided values (provided values take precedence)
            file_settings.update(values)
            values = file_settings

        # Load environment variables (returns parsed values for model, raw for cache)
        env_parsed, env_raw = self._load_env_settings()
        # Merge env settings (they take precedence over file settings)
        # Use deep_update with copy=True to avoid modifying the original dicts
        # (this is important when validation creates temporary instances)
        values = copy_module.deepcopy(values)
        deep_update(values, env_parsed, copy=False)

        super().__init__(**values)
        # Set private attributes after super().__init__()
        self._config_path = _cfg
        self._env_settings_cache = env_raw.copy()
        # Store the original file settings for later reference
        self._config_file_settings = original_file_settings

    def _load_env_settings(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load settings from environment variables.

        Supports both flat (NAPARI_FIELD) and nested (NAPARI_SECTION_FIELD) paths,
        as well as custom env names defined in field json_schema_extra.

        Returns
        -------
        tuple[dict, dict]
            A tuple of (parsed_values, raw_values). parsed_values are for model
            initialization, raw_values are for caching (to identify env-provided settings).
        """
        import json

        parsed: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        env_prefix = getattr(type(self), '_env_prefix', 'NAPARI_').upper()

        env_vars: Mapping[str, str | None] = {
            k.upper(): v for k, v in os.environ.items()
        }

        # Check for direct field mappings (flat access)
        # Use type(self).model_fields to avoid deprecation warning in V2.11+
        for field_name, field_info in type(self).model_fields.items():
            env_name = f'{env_prefix}{field_name.upper()}'
            if env_name in env_vars:
                val = env_vars[env_name]
                if val is not None:
                    raw[field_name] = val
                    # Try to parse as JSON for complex types
                    try:
                        parsed[field_name] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        parsed[field_name] = val

            # Check for nested field access (e.g., NAPARI_APPEARANCE_THEME)
            nested_prefix = f'{env_prefix}{field_name.upper()}_'
            for env_name, env_val in env_vars.items():
                if env_name.startswith(nested_prefix) and env_val is not None:
                    nested_path = env_name[len(nested_prefix) :].lower()
                    if field_name not in parsed:
                        parsed[field_name] = {}
                        raw[field_name] = {}
                    elif not isinstance(parsed[field_name], dict):
                        continue  # Already set to non-dict value
                    raw[field_name][nested_path] = env_val
                    # Try to parse nested value as JSON
                    try:
                        parsed[field_name][nested_path] = json.loads(env_val)
                    except (json.JSONDecodeError, TypeError):
                        parsed[field_name][nested_path] = env_val

            # Check for custom env names in nested model fields (json_schema_extra)
            annotation = field_info.annotation
            if annotation is not None:
                try:
                    if hasattr(annotation, 'model_fields'):
                        for (
                            nested_name,
                            nested_info,
                        ) in annotation.model_fields.items():
                            extra = nested_info.json_schema_extra or {}
                            if isinstance(extra, dict) and 'env' in extra:
                                custom_env = extra['env'].upper()
                                if custom_env in env_vars:
                                    val = env_vars[custom_env]
                                    if val is None:
                                        continue
                                    if field_name not in parsed:
                                        parsed[field_name] = {}
                                        raw[field_name] = {}
                                    elif not isinstance(
                                        parsed[field_name], dict
                                    ):
                                        continue
                                    raw[field_name][nested_name] = val
                                    # Try to parse as JSON, but handle booleans specially
                                    if val.lower() in ('true', '1', 'yes'):
                                        parsed[field_name][nested_name] = True
                                    elif val.lower() in ('false', '0', 'no'):
                                        parsed[field_name][nested_name] = False
                                    else:
                                        try:
                                            parsed[field_name][nested_name] = (
                                                json.loads(val)
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            TypeError,
                                        ):
                                            parsed[field_name][nested_name] = (
                                                val
                                            )
                except (TypeError, AttributeError):
                    pass

        return parsed, raw

    def _maybe_save(self):
        if self._save_on_change and self.config_path:
            self.save()

    def _on_sub_event(self, event, field=None):
        super()._on_sub_event(event, field)
        self._maybe_save()

    @property
    def config_path(self):
        """Return the path to/from which settings be saved/loaded."""
        return self._config_path

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: str = 'python',
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
        **kwargs: Any,
    ) -> DictStrAny:
        """Return dict representation of the model.

        May optionally specify which fields to include or exclude.
        """
        # Don't pass exclude_defaults to super() - we handle it ourselves
        # because Pydantic V2's exclude_defaults doesn't work correctly for
        # EventedModel (it compares private attrs which are always different)
        data = super().model_dump(
            mode=mode,
            include=include,  # type: ignore[arg-type]
            exclude=exclude,  # type: ignore[arg-type]
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=False,  # Handle below
            exclude_none=exclude_none,
            **kwargs,
        )
        if exclude_defaults:
            data = _exclude_defaults_evented(self, data)
        if exclude_env:
            self._remove_env_settings(data)
        return data

    # Backwards compatibility alias
    def dict(  # type: ignore[override]
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
    ) -> DictStrAny:
        """Return dict representation of the model (deprecated, use model_dump)."""
        return self.model_dump(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_env=exclude_env,
        )

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

    def save(self, path: str | Path | None = None, **dict_kwargs):
        """Save current settings to path.

        By default, this will exclude settings values that match the default
        value, and will exclude values that were provided by environment
        variables.  (see `_save_dict` method.)
        """
        path = path or self.config_path
        if not path:
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
            _data = self._yaml_dump(data)
        elif str(path).endswith('.json'):
            import json

            _data = json.dumps(data, default=str)
        else:
            raise NotImplementedError(
                trans._(
                    'Can only currently dump to `.json` or `.yaml`, not {path!r}',
                    deferred=True,
                    path=path,
                )
            )
        with open(path, 'w') as target:
            target.write(_data)

    def env_settings(self) -> Dict[str, Any]:
        """Get a dict of fields that were provided as environment vars."""
        return self._env_settings_cache

    def _remove_env_settings(self, data):
        """Remove key:values from `data` that match settings from env vars.

        This is handy when we want to persist settings to disk without
        including settings that were provided by environment variables (which
        are usually more temporary).
        """
        env_data = self.env_settings()
        if env_data:
            _restore_config_data(
                data, env_data, getattr(self, '_config_file_settings', {})
            )


# Utility functions


def config_file_settings_source(
    settings_cls: type,
    config_path: Path | str | None,
) -> dict[str, Any]:
    """Read config files during init of an EventedConfigFileSettings obj.

    Parameters
    ----------
    settings_cls : type
        The settings class
    config_path : Path | str | None
        Path to the config file

    Returns
    -------
    dict
        *validated* values for the model.
    """
    if not config_path:
        return {}

    sources: list[str] = []
    if config_path:
        sources.append(str(config_path))

    if not sources:
        return {}

    data: dict = {}
    for path in sources:
        if not path:
            continue
        path_ = Path(path).expanduser().resolve()

        # if the requested config path does not exist, move on to the next
        if not path_.is_file():
            continue

        # get loader for yaml/json
        if str(path).endswith(('.yaml', '.yml')):
            load = __import__('yaml').safe_load
        elif str(path).endswith('.json'):
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
        except Exception as err:
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

    try:
        # validate the data by creating a temporary instance
        settings_cls(config_path=None, **data)
    except ValidationError as err:
        # Check if strict mode is enabled - if so, re-raise the error
        if hasattr(settings_cls, 'model_config'):
            strict_check = settings_cls.model_config.get(
                'strict_config_check', False
            )
            if strict_check:
                raise

        # if errors occur, we still want to boot, so we just remove bad keys
        errors = err.errors()
        msg = trans._(
            'Validation errors in config file(s).\nThe following fields have been reset to the default value:\n\n{errors}\n',
            deferred=True,
            errors=display_errors(errors),
        )
        with contextlib.suppress(Exception):
            # we're about to nuke some settings, so just in case... try backup
            backup_path = path_.parent / f'{path_.stem}.BAK{path_.suffix}'
            backup_path.write_text(path_.read_text())

        _logger.warning(msg)
        try:
            _remove_bad_keys(data, [e.get('loc', ()) for e in errors])
        except KeyError:
            _logger.warning(
                trans._(
                    'Failed to remove validation errors from config file. Using defaults.'
                )
            )
            data = {}

    return data


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
            continue
        d = data
        while True:
            base, *key = key  # type: ignore[assignment]
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
            # Only recurse if the key exists in dct
            if k in dct:
                _restore_config_data(dct[k], v, dflt)
            elif dflt:
                # Key was excluded (e.g., by exclude_defaults), restore from defaults
                dct[k] = dflt
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

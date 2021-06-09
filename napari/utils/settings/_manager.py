"""Settings management.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional

from appdirs import user_config_dir
from yaml import safe_dump, safe_load

from ...utils.translations import trans
from .._base import _APPAUTHOR, _APPNAME, _FILENAME
from ._defaults import CORE_SETTINGS as CORE_SETTINGS
from ._defaults import (
    AppearanceSettings,
    ApplicationSettings,
    BaseNapariSettings,
    ExperimentalSettings,
    PluginsSettings,
    ShortcutsSettings,
)


class SettingsManager:
    """
    Napari settings manager using evented SettingsModels.

    This provides the presistence layer for the application settings.

    Parameters
    ----------
    config_path : str, optional
        Provide the base folder to store napari configuration. Default is None,
        which will point to user config provided by `appdirs`.
    save_to_disk : bool, optional
        Persist settings on disk. Default is True.

    Notes
    -----
    The settings manager will create a new user configuration folder which is
    provided by `appdirs` in a cross platform manner. On the first startup a
    new configuration file will be created using the default values defined by
    the `CORE_SETTINGS` models.

    If a configuration file is found in the specified location, it will be
    loaded by the `_load` method. On configuration load the following checks
    are performed:

    - If invalid sections are found, these will be removed from the file.
    - If invalid keys are found within a valid section, these will be removed
      from the file.
    - If invalid values are found within valid sections and valid keys, these
      will be replaced by the default value provided by `CORE_SETTINGS`
      models.
    """

    _FILENAME = _FILENAME
    _APPNAME = _APPNAME
    _APPAUTHOR = _APPAUTHOR
    appearance: AppearanceSettings
    application: ApplicationSettings
    plugins: PluginsSettings
    shortcuts: ShortcutsSettings
    experimental: ExperimentalSettings

    def __init__(
        self, config_path: Optional[str] = None, save_to_disk: bool = True
    ):
        self._config_path = (
            Path(user_config_dir(self._APPNAME, self._APPAUTHOR))
            if config_path is None
            else Path(config_path)
        )
        self._save_to_disk = save_to_disk
        self._settings: dict[str, BaseNapariSettings] = {}
        self._defaults: dict[str, BaseNapariSettings] = {}
        self._plugins: list = []
        self._env_settings: dict[str, Any] = {}

        if not self._config_path.is_dir():
            os.makedirs(self._config_path)

        self._load()

    def __getattr__(self, attr):
        if attr in self._settings:
            return self._settings[attr]

    def __dir__(self):
        """Add setting keys to make tab completion works."""
        return list(super().__dir__()) + list(self._settings)

    def __str__(self):
        return safe_dump(self._to_dict(safe=True))

    def _remove_default(self, settings_data):
        """
        Attempt to convert self to dict and to remove any default values from the configuration
        """

        for section, values in settings_data.items():
            if section not in self._defaults:
                continue

            default_values = self._defaults[section].dict()
            for k, v in list(values.items()):
                if default_values.get(k, None) == v:
                    del values[k]

        return settings_data

    def _to_dict(self, safe: bool = False) -> dict:
        """Convert the settings to a dictionary."""
        data = {}
        for section, model in self._settings.items():
            if safe:
                # We roundtrip to keep string objects (like SchemaVersion)
                # yaml representable
                data[section] = json.loads(model.json())
            else:
                data[section] = model.dict()

        return data

    def _save(self):
        """Save configuration to disk."""
        if self._save_to_disk:
            path = self.path / self._FILENAME

            if self._env_settings:
                # If using environment variables do not save them in the
                # `settings.yaml` file. We will use the original values found
                # in the file.
                loaded_data = BaseNapariSettings._LOADED_DATA
                data = self._to_dict(safe=True)
                for section, env_data in self._env_settings.items():
                    for env_key, _ in env_data.items():
                        try:
                            data[section][env_key] = loaded_data[section][
                                env_key
                            ]
                        except KeyError:
                            pass

                data_str = safe_dump(self._remove_default(data))
            else:
                data_str = safe_dump(
                    self._remove_default(self._to_dict(safe=True))
                )

            with open(path, "w") as fh:
                fh.write(data_str)

    def _load(self):
        """Read configuration from disk."""
        path = self.path / self._FILENAME

        if path.is_file():
            try:
                with open(path) as fh:
                    data = safe_load(fh.read()) or {}
            except Exception as err:
                warnings.warn(
                    trans._(
                        "The content of the napari settings file could not be read\n\nThe default settings will be used and the content of the file will be replaced the next time settings are changed.\n\nError:\n{err}",
                        deferred=True,
                        err=err,
                    )
                )
                data = {}

            # Load data once and save it in the base class
            BaseNapariSettings._LOADED_DATA = data

        for setting in CORE_SETTINGS:
            section = setting.schema().get("section", None)
            if section is None:
                raise ValueError(
                    trans._(
                        "Settings model {setting!r} must provide a `section` in the `schemas_extra`",
                        deferred=True,
                        setting=setting,
                    )
                )

            _section_defaults = {}
            for option, option_data in setting.schema()["properties"].items():
                _section_defaults[option] = option_data.get("default", None)

            self._defaults[section] = setting(**_section_defaults)
            model = setting()
            model.events.connect(lambda x: self._save())
            self._settings[section] = model
            self._env_settings[section] = getattr(
                model.__config__, "_env_settings"
            )(model)

        self._save()

    @property
    def path(self):
        return self._config_path

    def reset(self):
        """Reset settings to default values."""
        for section in self._settings:
            for key, default_value in self._defaults[section].dict().items():
                setattr(self._settings[section], key, default_value)

        self._save()

    def schemas(self) -> dict:
        """Return the json schema for each of the settings model."""
        schemas = {}
        for section, settings in self._settings.items():
            schemas[section] = {
                "json_schema": settings.schema_json(),
                "model": settings,
            }

        return schemas

    def register_plugin(self, plugin):
        """Register plugin settings with the settings manager.

        Parameters
        ----------
        plugin
            The napari plugin that may or may not provide settings.
        """
        self._plugins.append(plugin)


SETTINGS = SettingsManager()

"""Settings management.
"""

import os
from pathlib import Path

from pydantic import ValidationError
from yaml import safe_dump, safe_load

from napari.utils.settings._defaults import CORE_SETTINGS


class SettingsManager:
    """
    Napari settings manager using evented SettingsModels.

    This provides the presistence layer for the application settings.

    Parameters
    ----------
    config_path : str, optional
        Provide the base folder to store napari configuration. Default is None,
        which will point to `"~/.napari/settings/`.
    save_to_disk : bool, optional
        Persist settings on disk. Default is True.
    """

    _FILENAME = "settings.yaml"

    def __init__(self, config_path: str = None, save_to_disk: bool = True):
        # TODO: Use this instead Path.home() / ".config" / ".napari" ?
        self._config_path = (
            Path.home() / ".napari" / "settings"
            if config_path is None
            else config_path
        )
        self._save_to_disk = save_to_disk
        self._settings = {}
        self._defaults = {}
        self._models = {}
        self._plugins = []

        if not self._config_path.is_dir():
            os.makedirs(self._config_path)

        self._load()

    def __getattr__(self, attr):
        if attr in self._settings:
            return self._settings[attr]

    def __dir__(self):
        """Add setting keys to make tab completion works."""
        return super().__dir__() + list(self._settings)

    @staticmethod
    def _get_section_name(settings) -> str:
        """
        Return the normalized name of a section based on its config title.
        """
        section = settings.Config.title.replace(" ", "_").lower()
        if section.endswith("_settings"):
            section = section.replace("_settings", "")

        return section

    def _to_dict(self) -> dict:
        """Convert the settings to a dictionary."""
        data = {}
        for section, model in self._settings.items():
            data[section] = model.dict()

        return data

    def _save(self):
        """Save configuration to disk."""
        if self._save_to_disk:
            path = self._config_path / self._FILENAME
            with open(path, "w") as fh:
                fh.write(safe_dump(self._to_dict()))

    def _load(self):
        """Read configuration from disk.

        If no settings file is found, a new one is created using the defaults.
        """
        path = self._config_path / self._FILENAME
        for plugin in CORE_SETTINGS:
            section = self._get_section_name(plugin)
            self._defaults[section] = plugin()
            self._models[section] = plugin

        if path.is_file():
            with open(path) as fh:
                data = safe_load(fh.read())

            # Check with models
            for section, model_data in data.items():
                try:
                    model = self._models[section](**model_data)
                    model.events.connect(lambda x: self._save())
                    self._settings[section] = model
                except KeyError:
                    pass
                except ValidationError as e:
                    # Handle extra fields
                    model_data_replace = {}
                    for error in e.errors():
                        # Grab the first error entry
                        item = error["loc"][0]
                        try:
                            model_data_replace[item] = getattr(
                                self._defaults[section], item
                            )
                        except AttributeError:
                            model_data.pop(item)

                    model_data.update(model_data_replace)
                    model = self._models[section](**model_data)
                    model.events.connect(lambda x: self._save())
                    self._settings[section] = model
        else:
            self._settings = self._defaults

        self._save()

    def reset(self):
        """Reset settings to default values."""
        for section in self._settings:
            self._settings[section] = self._models[section]()

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
        plugin:
            The napari plugin that may or may not provide settings.
        """
        self._plugins.append(plugin)


SETTINGS = SettingsManager()

"""Settings management.
"""

import os
from pathlib import Path

from pydantic import ValidationError
from yaml import safe_dump, safe_load

from napari.utils.settings._defaults import (
    ApplicationSettings,
    ConsoleSettings,
    PluginSettings,
)


class SettingsManager:
    """"""

    def __init__(self, config_path=None):
        self._config_path = Path.home() / ".napari" / "settings"
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

    def _get_section_name(self, settings):
        """
        Return the normalized name of a section based on its config or class name.
        """
        section = settings.Config.title.replace(" ", "_").lower()
        if section.endswith("_settings"):
            section = section.replace("_settings", "")

        return section

    def _save(self):
        """Save configuration to disk."""
        data = {}
        for section, model in self._settings.items():
            data[section] = model.dict()

        path = self._config_path / "settings.yaml"
        with open(path, "w") as fh:
            fh.write(safe_dump(data))

    def _load(self):
        """Read configuration from disk.

        If no settings file is found, a new one is created using the defaults.
        """
        path = self._config_path / "settings.yaml"
        for plugin in [ApplicationSettings, ConsoleSettings, PluginSettings]:
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
                    # FIXME: Handle extra fields. add a field that does not exist in the model
                    model_data_replace = {}
                    for error in e.errors():
                        item = error["loc"][
                            0
                        ]  # FIXME: For now grab the first item
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

    def schemas(self):
        """Return the json schema for each of the settings model."""
        schemas = {}
        for section, settings in self._settings.items():
            schemas[section] = {
                "json_schema": settings.json_schema(),
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

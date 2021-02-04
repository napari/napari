"""Settings management.
"""

import os
from json import dumps
from pathlib import Path

import toml
from pydantic import ValidationError
from yaml import safe_dump, safe_load

from napari.utils.settings._defaults import (
    ApplicationSettings,
    ConsoleSettings,
)


class SettingsManager:
    """"""

    def __init__(self, config_path=None):
        self._config_path = Path.home() / ".napari" / "settings"
        self._settings = {}
        self._defaults = {}
        self._models = {}
        self._plugins = []
        self._format = "yaml"

        if not self._config_path.is_dir():
            os.makedirs(self._config_path)

        self._load()

    def __getattr__(self, attr):
        if attr in self._settings:
            return self._settings[attr]

    def _get_file_name(self, settings):
        """"""
        key = settings.Config.title.replace(" ", "_").lower()
        if key.endswith("_settings"):
            key = key.replace("_settings", "")

        return key

    def _save(self):
        """Save configuration to disk."""
        data = {}
        for key, model in self._settings.items():
            data[key] = model.dict()

        path = self._config_path / f"settings.{self._format}"
        with open(path, "w") as fh:
            fh.write(safe_dump(data))

        path = self._config_path / "settings.json"
        with open(path, "w") as fh:
            fh.write(dumps(data, sort_keys=True, indent=4))

        path = self._config_path / "settings.toml"
        with open(path, "w") as fh:
            fh.write(toml.dumps(data))

    def _load(self):
        """Read configuration from disk.

        If no settings file is found, a new one is created using the defaults.
        """
        path = self._config_path / f"settings.{self._format}"
        for plugin in [ApplicationSettings, ConsoleSettings]:
            key = self._get_file_name(plugin)
            self._defaults[key] = plugin()
            self._models[key] = plugin

        if path.is_file():
            with open(path) as fh:
                data = safe_load(fh.read())

            # Check with models
            for key, model_data in data.items():
                try:
                    model = self._models[key](**model_data)
                    model.events.connect(lambda x: self._save())
                    self._settings[key] = model
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
                                self._defaults[key], item
                            )
                        except AttributeError:
                            model_data.pop(item)

                    model_data.update(model_data_replace)
                    # FIXME: Add a try except
                    model = self._models[key](**model_data)
                    model.events.connect(lambda x: self._save())
                    self._settings[key] = model
        else:
            self._settings = self._defaults

        self._save()

    def schemas(self):
        """Return the json schema for each of the settings model."""
        schemas = {}
        for key, settings in self._settings.items():
            schemas[key] = {
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

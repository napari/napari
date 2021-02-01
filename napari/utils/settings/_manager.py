"""Settings management.
"""


import os
from collections import OrderedDict
from enum import Enum
from json import dumps

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
        self._config_path = os.path.join(
            os.path.expanduser("~/"), ".napari", "settings"
        )
        self._settings = {}
        self._defaults = {}
        self._models = {}
        self._plugins = []
        self._format = "yaml"

        if not os.path.isdir(self._config_path):
            os.makedirs(self._config_path)

        self._load()

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

        path = os.path.join(self._config_path, f"settings.{self._format}")
        with open(path, "w") as fh:
            fh.write(safe_dump(data))

        path = os.path.join(self._config_path, "settings.json")
        with open(path, "w") as fh:
            fh.write(dumps(data, sort_keys=True, indent=4))

        path = os.path.join(self._config_path, "settings.toml")
        with open(path, "w") as fh:
            fh.write(toml.dumps(data))

    def _load(self):
        """Read configuration from disk.

        If no settings file is found, a new one is created using the defaults.
        """
        path = os.path.join(self._config_path, f"settings.{self._format}")
        for plugin in [ApplicationSettings, ConsoleSettings]:
            key = self._get_file_name(plugin)
            self._defaults[key] = plugin()
            self._models[key] = plugin

        if os.path.isfile(path):
            with open(path) as fh:
                data = safe_load(fh.read())

            # Check with models
            for key, model_data in data.items():
                try:
                    self._settings[key] = self._models[key](**model_data)
                except KeyError as e:
                    pass
                except ValidationError as e:
                    # FIXME: Handle extra fields. add a field that does not exist in the model
                    model_data_replace = {}
                    for error in e.errors():
                        item = error["loc"][
                            0
                        ]  # FIXME: For now grab the frist item
                        model_data_replace[item] = getattr(
                            self._defaults[key], item
                        )

                    model_data.update(model_data_replace)
                    # FIXME: Add a try except
                    self._settings[key] = self._models[key](**model_data)
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

    def setting(self, key):
        """"""
        settings = None

        if key in self._settings:
            settings = self._settings[key]

        return settings

    def register_plugin(self, plugin):
        """Register plugin settings with the settings manager.

        Parameters
        ----------
        plugin:
            The napari plugin that may or may not provide settings.
        """
        self._plugins.append(plugin)

    def get(self, section, option):
        """"""
        return getattr(self._settings[section], option)

    def set(self, section, option, value):
        """

        Parametes
        ---------
        section: str
            TODO:
        option: str
            TODO:
        value:
            TODO:
        """
        setattr(self._settings[section], option, value)
        self._save()  # Expensive!


SETTINGS = SettingsManager()

import os
from pathlib import Path
from typing import Optional

from .._base import _DEFAULT_CONFIG_PATH
from ._appearance import AppearanceSettings
from ._application import ApplicationSettings
from ._base import BaseNapariSettings
from ._experimental import ExperimentalSettings
from ._plugins import PluginsSettings
from ._shortcuts import ShortcutsSettings

_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)


class NapariSettings(BaseNapariSettings):
    application: ApplicationSettings
    appearance: AppearanceSettings
    plugins: PluginsSettings
    shortcuts: ShortcutsSettings
    experimental: ExperimentalSettings

    # private attributes and ClassVars will not appear in the schema
    _config_path: Optional[Path] = Path(_CFG_PATH)

    class Config:
        env_prefix = 'napari_'

    def __str__(self):
        out = 'NapariSettings (defaults excluded)\n'
        out += '----------------------------------\n'
        out += self.yaml(exclude_defaults=True)
        return out

    def __repr__(self):
        return str(self)

    # TODO: remove legacy function
    def schemas(self) -> dict:
        """Return the json schema for each of the settings model."""
        return {
            name: {
                "json_schema": field.type_.schema_json(),
                "model": getattr(self, name),
            }
            for name, field in self.__fields__.items()
        }

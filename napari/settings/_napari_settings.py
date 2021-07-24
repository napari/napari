import os
from pathlib import Path
from typing import Optional

from pydantic import Field

from ..utils._base import _DEFAULT_CONFIG_PATH
from ._appearance import AppearanceSettings
from ._application import ApplicationSettings
from ._base import EventedConfigFileSettings
from ._experimental import ExperimentalSettings
from ._plugins import PluginsSettings
from ._shortcuts import ShortcutsSettings

_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)


class NapariSettings(EventedConfigFileSettings):
    application: ApplicationSettings = Field(
        default_factory=ApplicationSettings
    )
    appearance: AppearanceSettings = Field(default_factory=AppearanceSettings)
    plugins: PluginsSettings = Field(default_factory=PluginsSettings)
    shortcuts: ShortcutsSettings = Field(default_factory=ShortcutsSettings)
    experimental: ExperimentalSettings = Field(
        default_factory=ExperimentalSettings
    )

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

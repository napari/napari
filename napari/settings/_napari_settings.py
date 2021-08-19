import os
from pathlib import Path
from typing import Optional

from pydantic import Field

from ..utils._base import _DEFAULT_CONFIG_PATH
from ..utils.translations import trans
from ._appearance import AppearanceSettings
from ._application import ApplicationSettings
from ._base import EventedConfigFileSettings
from ._experimental import ExperimentalSettings
from ._plugins import PluginsSettings
from ._shortcuts import ShortcutsSettings

_DEFAULT_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)
_DEFAULT_CFG_PATH = Path(_DEFAULT_CFG_PATH) if _DEFAULT_CFG_PATH else None


class NapariSettings(EventedConfigFileSettings):
    """Schema for napari settings."""

    application: ApplicationSettings = Field(
        default_factory=ApplicationSettings,
        title=trans._("Application"),
        description=trans._("Main application settings."),
    )
    appearance: AppearanceSettings = Field(
        default_factory=AppearanceSettings,
        title=trans._("Appearance"),
        description=trans._("User interface appearance settings."),
    )
    plugins: PluginsSettings = Field(
        default_factory=PluginsSettings,
        title=trans._("Plugins"),
        description=trans._("Plugins settings."),
    )
    shortcuts: ShortcutsSettings = Field(
        default_factory=ShortcutsSettings,
        title=trans._("Shortcuts"),
        description=trans._("Shortcut settings."),
    )
    experimental: ExperimentalSettings = Field(
        default_factory=ExperimentalSettings,
        title=trans._("Experimental"),
        description=trans._("Experimental settings."),
    )

    # private attributes and ClassVars will not appear in the schema
    _config_path: Optional[Path] = _DEFAULT_CFG_PATH

    class Config:
        env_prefix = 'napari_'
        use_enum_values = False
        # all of these fields are evented models, so we don't want to break
        # connections by setting the top-level field itself
        # (you can still mutate attributes in the subfields)
        allow_mutation = False

    def __str__(self):
        out = 'NapariSettings (defaults excluded)\n'
        out += '----------------------------------\n'
        out += self._yaml_dump(self._save_dict())
        return out

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 2:
        dest = Path(sys.argv[2]).expanduser().absolute()
    else:
        dest = Path(__file__).parent / 'napari.schema.json'
    dest.write_text(NapariSettings.schema_json())

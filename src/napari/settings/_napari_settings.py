import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from napari.settings._appearance import AppearanceSettings
from napari.settings._application import ApplicationSettings
from napari.settings._experimental import ExperimentalSettings
from napari.settings._fields import Version
from napari.settings._general_settings import GeneralSettings
from napari.settings._plugins import PluginsSettings
from napari.settings._shortcuts import ShortcutsSettings
from napari.utils._base import _DEFAULT_CONFIG_PATH

_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)

CURRENT_SCHEMA_VERSION = Version(0, 9, 0)


class NapariSettings(GeneralSettings):
    """Schema for napari settings."""

    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option

    application: ApplicationSettings = Field(
        default_factory=ApplicationSettings,
        title='Application',
        description='Main application settings.',
    )
    appearance: AppearanceSettings = Field(
        default_factory=AppearanceSettings,
        title='Appearance',
        description='User interface appearance settings.',
        frozen=True,
    )
    plugins: PluginsSettings = Field(
        default_factory=PluginsSettings,
        title='Plugins',
        description='Plugins settings.',
        frozen=True,
    )
    shortcuts: ShortcutsSettings = Field(
        default_factory=ShortcutsSettings,
        title='Shortcuts',
        description='Shortcut settings.',
        frozen=True,
    )
    experimental: ExperimentalSettings = Field(
        default_factory=ExperimentalSettings,
        title='Experimental',
        description='Experimental settings.',
        frozen=True,
    )

    # private attributes and ClassVars will not appear in the schema
    config_path: Path | None = Field(
        Path(_CFG_PATH) if _CFG_PATH else None, exclude=True
    )

    model_config = SettingsConfigDict(
        env_prefix='napari_',
        nested_model_default_partial_update=True,
        env_nested_delimiter='_',
        env_nested_max_split=1,
        use_enum_values=False,
        extra='ignore',
        populate_by_name=True,
    )


if __name__ == '__main__':
    import json
    import sys

    if len(sys.argv) > 2:
        dest = Path(sys.argv[2]).expanduser().absolute()
    else:
        dest = Path(__file__).parent / 'napari.schema.json'
    dest.write_text(json.dumps(NapariSettings.model_json_schema()))

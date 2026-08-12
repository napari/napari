from __future__ import annotations

import os
from pathlib import Path

from pydantic import (
    Field,
)
from pydantic_settings import (
    SettingsConfigDict,
)

from napari.settings._base import (
    EventedConfigFileSettings,
    _NotSetType,
    _remove_empty_dicts,
)
from napari.utils._platformdirs import user_config_dir

_PL_CFG_PATH = os.getenv('NAPARI_CONFIG', user_config_dir())


class PluginPreferences(EventedConfigFileSettings):
    model_config = SettingsConfigDict(
        env_prefix='napari_',
        nested_model_default_partial_update=True,
        env_nested_delimiter='_',
        env_nested_max_split=1,
        use_enum_values=False,
        extra='ignore',
        populate_by_name=True,
    )
    config_path: Path | _NotSetType | None = Field(
        Path(_PL_CFG_PATH) if _PL_CFG_PATH else None, exclude=True
    )

    def __str__(self) -> str:
        out = 'PluginSettings (defaults excluded)\n' + 34 * '-' + '\n'
        data = self.model_dump(exclude_defaults=True)
        out += self._yaml_dump(_remove_empty_dicts(data))
        return out

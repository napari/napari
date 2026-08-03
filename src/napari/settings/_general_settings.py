import os
from abc import ABC
from typing import Any

from pydantic import Field

from napari.settings._base import (
    _NOT_SET,
    EventedConfigFileSettings,
    _remove_empty_dicts,
)
from napari.settings._fields import Version
from napari.utils._base import _DEFAULT_CONFIG_PATH

_CFG_PATH = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)

CURRENT_SCHEMA_VERSION = Version(0, 9, 0)


class GeneralSettings(EventedConfigFileSettings, ABC):
    """Schema for settings."""

    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Version = Field(
        CURRENT_SCHEMA_VERSION,
        description='Napari settings schema version.',
    )

    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        super().__init__(config_path, **values)
        self._maybe_migrate()

    def _save_dict(self, **kwargs):
        # we always want schema_version written to the settings.yaml
        # TODO: is there a better way to always include schema version?
        return {
            'schema_version': self.schema_version,
            **super()._save_dict(**kwargs),
        }

    def __str__(self):
        out = 'NapariSettings (defaults excluded)\n' + 34 * '-' + '\n'
        data = self.model_dump(exclude_defaults=True)
        out += self._yaml_dump(_remove_empty_dicts(data))
        return out

    def __repr__(self):
        return str(self)

    def _maybe_migrate(self):
        if self.schema_version < CURRENT_SCHEMA_VERSION:
            from napari.settings._migrations import do_migrations

            do_migrations(self)

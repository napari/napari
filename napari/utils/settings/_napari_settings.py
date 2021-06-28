import os
from typing import Any

from .._base import _DEFAULT_CONFIG_PATH
from ..events import EmitterGroup
from ._appearance import AppearanceSettings
from ._application import ApplicationSettings
from ._base import BaseNapariSettings
from ._experimental import ExperimentalSettings
from ._plugins import PluginsSettings
from ._shortcuts import ShortcutsSettings


class NapariSettings(BaseNapariSettings):
    application: ApplicationSettings
    appearance: AppearanceSettings
    plugins: PluginsSettings
    shortcuts: ShortcutsSettings
    experimental: ExperimentalSettings

    _save_on_change: bool = True

    class Config:
        env_prefix = 'napari_'
        load_from = [os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)]
        save_to = os.getenv('NAPARI_CONFIG', _DEFAULT_CONFIG_PATH)

    # TODO: use config_path from init
    def __init__(self, config_path=None, **values: Any) -> None:
        super().__init__(**values)

        # look for eventedModel subfields and connect to self.save
        for name in self.__fields__:
            attr = getattr(self, name)
            if isinstance(getattr(attr, 'events', None), EmitterGroup):
                attr.events.connect(self._on_sub_event)

    def _on_sub_event(self, event):
        # TODO: re-emit event so listeners can watch just this.
        if self._save_on_change:
            self.save()

    def __str__(self):
        out = 'NapariSettings (defaults excluded)\n'
        out += '----------------------------------\n'
        out += self.yaml(exclude_defaults=True)
        return out

    def __repr__(self):
        return str(self)

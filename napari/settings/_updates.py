from enum import Enum
from pydantic import Field

from ..utils.events.evented_model import EventedModel
from ..utils.translations import trans


class UpdateOn(str, Enum):
    closing = 'closing'
    opening = 'opening'


class UpdateSettings(EventedModel):
    check_for_updates: bool = Field(
        True,
        title=trans._("Automatically check for napari updates"),
        description=trans._("Automatically check for napari updates."),
    )
    update_to_latest: bool = Field(
        True,
        title=trans._("Update to the latest version automatically"),
        description=trans._("Update to the latest version automatically."),
    )
    notify_update: bool = Field(
        True,
        title=trans._("Notify me before updating"),
        description=trans._("Notify me before updating."),
    )
    update_napari_on: UpdateOn = Field(
        UpdateOn.closing,
        title=trans._("Update napari upon"),
        description=trans._("Update napari upon.",),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

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
    check_previews = Field(
        False,
        title=trans._("Check for preview candidates "),
        description=trans._(
            "Check for napari non-stable release.",
        ),
    )
    update_version_skip = Field(
        [],
        title=trans._("Skip napari versions"),
        description=trans._(
            "Skip napari versions.",
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version', 'update_version_skip']

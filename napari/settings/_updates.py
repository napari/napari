from typing import List

from pydantic import Field

from ..utils.events.evented_model import EventedModel
from ..utils.misc import is_dev
from ..utils.translations import trans


class UpdateSettings(EventedModel):
    check_for_updates: bool = Field(
        True,
        title=trans._("Automatically check for napari updates"),
        description=trans._("Automatically check for napari updates."),
    )
    update_automatically: bool = Field(
        False,
        title=trans._("Update to the latest version automatically"),
        description=trans._("Update to the latest version automatically."),
    )
    check_previews: bool = Field(
        False,
        title=trans._("Check preview candidates "),
        description=trans._(
            "Check preview candidates.",
        ),
    )
    check_nightly_builds: bool = Field(
        False,
        title=trans._("Check nightly builds"),
        description=trans._("Check nightly builds."),
    )
    update_version_skip: List[str] = Field(
        [],
        title=trans._("Skip napari versions"),
        description=trans._(
            "Skip napari versions.",
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        if is_dev():
            preferences_exclude = ['schema_version', 'update_version_skip']
        else:
            preferences_exclude = [
                'schema_version',
                'update_version_skip',
                'check_nightly_builds',
                'check_previews',
            ]

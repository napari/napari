from typing import Tuple, Union

from pydantic import Field

from ..utils.events.evented_model import EventedModel
from ..utils.translations import trans
from ._fields import SchemaVersion, Theme


class AppearanceSettings(EventedModel):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 1)

    theme: Theme = Field(
        "dark",
        title=trans._("Theme"),
        description=trans._("Select the user interface theme."),
        env="napari_theme",
    )

    highlight_thickness: int = Field(
        1,
        title=trans._("Highlight thickness"),
        description=trans._(
            "Select the highlight thickness when hovering over shapes/points."
        ),
        ge=1,
        le=10,
    )

    layer_tooltip_visibility: bool = Field(
        False,
        title=trans._("Show layer tooltips"),
        description=trans._("Toggle to display a tooltip on mouse hover."),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

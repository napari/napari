from typing import Tuple, Union

from pydantic import Field

from ..utils.translations import trans
from ._base import EventedSettings
from ._fields import SchemaVersion

# this class inherits from EventedSettings instead of EventedModel because
# it uses Field(env=...) for one of its attributes


class ExperimentalSettings(EventedSettings):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 1, 0)
    octree: Union[bool, str] = Field(
        False,
        title=trans._("Enable Asynchronous Tiling of Images"),
        description=trans._(
            "Renders images asynchronously using tiles. \nYou must restart napari for changes of this setting to apply."
        ),
        type='boolean',  # need to specify to build checkbox in preferences.
        requires_restart=True,
    )

    async_: bool = Field(
        False,
        title=trans._("Render Images Asynchronously"),
        description=trans._(
            "Asynchronous loading of image data. \nThis setting partially loads data while viewing. \nYou must restart napari for changes of this setting to apply."
        ),
        env="napari_async",
        requires_restart=True,
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

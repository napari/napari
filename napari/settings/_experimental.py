from typing import Union

from pydantic import Field

from napari.settings._base import EventedSettings
from napari.utils.translations import trans


# this class inherits from EventedSettings instead of EventedModel because
# it uses Field(env=...) for one of its attributes
class ExperimentalSettings(EventedSettings):
    async_: bool = Field(
        False,
        title=trans._("Render Images Asynchronously"),
        description=trans._(
            "Asynchronous loading of image data. \nThis setting partially loads data while viewing. \nYou must restart napari for changes of this setting to apply."
        ),
        env="napari_async",
        requires_restart=True,
    )
    octree: Union[bool, str] = Field(
        False,
        title=trans._("Legacy octree setting (deprecated)"),
        description=trans._(
            "Renders images asynchronously using tiles. \nYou must restart napari for changes of this setting to apply."
        ),
        type='boolean',  # need to specify to build checkbox in preferences.
        requires_restart=True,
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

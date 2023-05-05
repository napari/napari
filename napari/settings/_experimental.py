from typing import Union

from pydantic import Field

from napari.settings._base import EventedSettings
from napari.utils.translations import trans


# this class inherits from EventedSettings instead of EventedModel because
# it uses Field(env=...) for one of its attributes
class ExperimentalSettings(EventedSettings):
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
    autoswap_buffers: bool = Field(
        False,
        title=trans._("Enable autoswapping rendering buffers."),
        description=trans._(
            "Autoswapping rendering buffers improves quality by reducing tearing artifacts, while sacrificing some performance."
        ),
        env="napari_autoswap",
        requires_restart=True,
    )

    rdp_epsilon: float = Field(
        0.5,
        title=trans._("RDP epsilon"),
        description=trans._(
            "Setting this higher removes more points from polygons. \nSetting this to 0 keeps all vertices of a given polygon"
        ),
        type=float,
        ge=0,
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

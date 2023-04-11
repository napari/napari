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
            "Asynchronous loading of image data. \nThis setting partially loads data while viewing."
        ),
        env="napari_async",
        requires_restart=False,
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ['schema_version']

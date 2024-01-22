from napari._pydantic_compat import Field
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
        title=trans._("Shapes polygon lasso RDP epsilon"),
        description=trans._(
            "Setting this higher removes more points from polygons. \nSetting this to 0 keeps all vertices of a given polygon"
        ),
        type=float,
        ge=0,
    )

    lasso_vertex_distance: int = Field(
        10,
        title=trans._("Minimum distance threshold of shapes lasso tool"),
        description=trans._(
            "Value determines how many screen pixels one has to move before another vertex can be added to the polygon."
        ),
        type=int,
        gt=0,
        lt=50,
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)

from typing import Any

from napari._pydantic_compat import Field
from napari.settings._base import EventedSettings
from napari.utils.translations import trans
from napari.utils.triangulation_backend import TriangulationBackend


# this class inherits from EventedSettings instead of EventedModel because
# it uses Field(env=...) for one of its attributes
class ExperimentalSettings(EventedSettings):
    def __init__(self, **data: dict[str, Any]):
        super().__init__(**data)

        self.events.triangulation_backend.connect(
            _update_triangulation_backend
        )
        self.events.triangulation_backend(value=self.triangulation_backend)

    async_: bool = Field(
        False,
        title=trans._('Render Images Asynchronously'),
        description=trans._(
            'Asynchronous loading of image data. \nThis setting partially loads data while viewing.'
        ),
        env='napari_async',
        requires_restart=False,
    )
    autoswap_buffers: bool = Field(
        False,
        title=trans._('Enable autoswapping rendering buffers.'),
        description=trans._(
            'Autoswapping rendering buffers improves quality by reducing tearing artifacts, while sacrificing some performance.'
        ),
        env='napari_autoswap',
        requires_restart=True,
    )

    rdp_epsilon: float = Field(
        0.5,
        title=trans._('Shapes polygon lasso and path RDP epsilon'),
        description=trans._(
            'Setting this higher removes more points from polygons or paths. \nSetting this to 0 keeps all vertices of '
            'a given polygon or path.'
        ),
        type=float,
        ge=0,
    )

    lasso_vertex_distance: int = Field(
        10,
        title=trans._(
            'Minimum distance threshold of shapes lasso and path tool'
        ),
        description=trans._(
            'Value determines how many screen pixels one has to move before another vertex can be added to the polygon'
            'or path.'
        ),
        type=int,
        gt=0,
        lt=50,
    )

    completion_radius: int = Field(
        default=-1,
        title=trans._(
            'Double-click Labels polygon completion radius (-1 to always complete)'
        ),
        description=trans._(
            'Max radius in pixels from first vertex for double-click to complete a polygon; set -1 to always complete.'
        ),
    )
    triangulation_backend: TriangulationBackend = Field(
        TriangulationBackend.none,
        title=trans._('Triangulation backend to use for Shapes layer'),
        description=trans._(
            'Triangulation backend to use for Shapes layer.\n'
            "The 'bermuda' requires the optional 'bermuda' package.\n"
            "The 'partsegcore' requires the optional 'partsegcore-compiled-backend' package.\n"
            "The 'triangle' requires the optional 'triangle' package.\n"
            "The 'none' backend uses the default Python triangulation.\n"
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version',)


def _update_triangulation_backend(event) -> None:
    from napari.layers.shapes import _accelerated_triangulate_dispatch
    from napari.layers.shapes._shapes_models import shape

    experimental = event.source

    _accelerated_triangulate_dispatch.USE_COMPILED_BACKEND = (
        experimental.triangulation_backend
        in {TriangulationBackend.partsegcore, TriangulationBackend.bermuda}
    )
    shape.TRIANGULATION_BACKEND = experimental.triangulation_backend

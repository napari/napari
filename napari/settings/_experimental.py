from typing import Any

from napari._pydantic_compat import Field
from napari.settings._base import EventedSettings
from napari.utils.events import Event
from napari.utils.translations import trans
from napari.utils.triangulation_backend import (
    TriangulationBackend,
    set_backend,
)


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
        TriangulationBackend.fastest_available,
        title=trans._('Triangulation backend to use for Shapes layer'),
        description=trans._(
            'Triangulation backend to use for Shapes layer.\n'
            "The 'bermuda' requires the optional 'bermuda' package.\n"
            "The 'partsegcore' requires the optional 'partsegcore-compiled-backend' package.\n"
            "The 'triangle' requires the optional 'triangle' package.\n"
            "The 'numba' backend requires the optional 'numba' package.\n"
            "The 'pure python' backend uses the default Python triangulation from vispy.\n"
            "The 'fastest available' backend will select the fastest available backend.\n"
        ),
        env='napari_triangulation_backend',
    )

    compiled_triangulation: bool = Field(
        default=False,
        title=trans._('Unused option. Use "triangulation backend" instead.'),
        description=trans._(
            'This option was removed in napari 0.6.0. Use \n'
            '"triangulation backend" instead.'
        ),
    )

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = ('schema_version', 'compiled_triangulation')


def _update_triangulation_backend(event: Event) -> None:
    experimental: ExperimentalSettings = event.source

    set_backend(experimental.triangulation_backend)

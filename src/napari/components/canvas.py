import numpy as np

from napari._pydantic_compat import Field, PrivateAttr
from napari.components.grid import GridCanvas
from napari.components.overlays import (
    BrushCircleOverlay,
    CanvasOverlay,
    CurrentSliceOverlay,
    ScaleBarOverlay,
    TextOverlay,
    WelcomeOverlay,
    ZoomOverlay,
)
from napari.settings import get_settings
from napari.utils.color import ColorValue
from napari.utils.compat import StrEnum
from napari.utils.events import EventedDict, EventedModel
from napari.utils.theme import get_theme

DEFAULT_CANVAS_OVERLAYS = {
    'welcome': WelcomeOverlay,
    'scale_bar': ScaleBarOverlay,
    'text': TextOverlay,
    'brush_circle': BrushCircleOverlay,
    'zoom': ZoomOverlay,
    'current_slice': CurrentSliceOverlay,
}


class Orientation(StrEnum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


class OverlayTiling(EventedModel):
    padding: tuple[float, float] = (10.0, 10.0)
    top_left: Orientation = Orientation.VERTICAL
    top_center: Orientation = Orientation.VERTICAL
    top_right: Orientation = Orientation.HORIZONTAL
    bottom_left: Orientation = Orientation.HORIZONTAL
    bottom_center: Orientation = Orientation.VERTICAL
    bottom_right: Orientation = Orientation.VERTICAL


class Canvas(EventedModel):
    """
    Canvas evented model.

    Controls canvas-related attributes, such as grid mode and canvas overlays.

    Attributes
    ----------
    background_color :
        ...
    grid :
        ...
    _canvas_size: Tuple[int, int]
        The canvas size following the Numpy convention of height x width
    """

    background_color_override: ColorValue | None = None
    grid: GridCanvas = Field(default_factory=GridCanvas, allow_mutation=False)
    overlay_tiling: OverlayTiling = Field(
        default_factory=OverlayTiling, allow_mutation=False
    )
    _overlays: EventedDict[str, CanvasOverlay] = PrivateAttr(
        default_factory=EventedDict
    )
    _size: tuple[int, int] = PrivateAttr((800, 600))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._update_viewer_grid()

        settings = get_settings()
        settings.application.events.grid_stride.connect(
            self._update_viewer_grid
        )
        settings.application.events.grid_width.connect(
            self._update_viewer_grid
        )
        settings.application.events.grid_height.connect(
            self._update_viewer_grid
        )
        settings.application.events.grid_spacing.connect(
            self._update_viewer_grid
        )

        self._overlays.update(
            {k: v() for k, v in DEFAULT_CANVAS_OVERLAYS.items()}
        )

    @property
    def size(self):
        return self._size

    def viewbox_size(self, n_layers):
        """Get the size of a single viewbox (whether grid is enabled or not).

        If grid.border_width > 0, that's accounted for too.
        """
        viewbox_size = np.array(self.size)
        if self.grid.enabled:
            grid_shape = np.array(self.grid.actual_shape(n_layers))
            spacing_pixels = self.grid._compute_canvas_spacing(
                self.size, n_layers
            )
            # Now calculate actual available space
            total_gap_space = spacing_pixels * (grid_shape - 1)
            available_space = self.size - total_gap_space
            viewbox_size = available_space / grid_shape
        return viewbox_size

    @property
    def scale_bar(self):
        return self._overlays['scale_bar']

    @property
    def text(self):
        return self._overlays['text']

    @property
    def welcome(self):
        return self._overlays['welcome']

    @property
    def _zoom_box(self):
        return self._overlays['zoom']

    @property
    def _brush_circle_overlay(self):
        return self._overlays['brush_circle']

    def _update_viewer_grid(self):
        """Keep viewer grid settings up to date with settings values."""

        settings = get_settings()

        self.grid.stride = settings.application.grid_stride
        self.grid.shape = (
            settings.application.grid_height,
            settings.application.grid_width,
        )
        self.grid.spacing = settings.application.grid_spacing

    @property
    def background_color(self):
        if self.background_color_override is not None:
            return self.background_color_override

        return ColorValue(
            get_theme(get_settings().appearance.theme).canvas.as_rgb_tuple()
        )

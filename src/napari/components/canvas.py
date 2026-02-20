from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr

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
    grid: GridCanvas = Field(default_factory=GridCanvas, frozen=True)
    overlay_tiling: OverlayTiling = Field(
        default_factory=OverlayTiling, frozen=True
    )
    _overlays: EventedDict[str, CanvasOverlay] = PrivateAttr(
        default_factory=EventedDict
    )
    _size: tuple[int, int] = PrivateAttr((800, 600))

    def __init__(self, **kwargs: Any) -> None:
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
    def size(self) -> tuple[int, int]:
        return self._size

    def viewbox_size(self, n_layers: int) -> tuple[int, int]:
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
        return tuple(viewbox_size)

    @property
    def scale_bar(self) -> ScaleBarOverlay:
        return self._overlays['scale_bar']  # type: ignore[return-value]

    @property
    def text(self) -> TextOverlay:
        return self._overlays['text']  # type: ignore[return-value]

    @property
    def welcome(self) -> WelcomeOverlay:
        return self._overlays['welcome']  # type: ignore[return-value]

    @property
    def _zoom_box(self) -> ZoomOverlay:
        return self._overlays['zoom']  # type: ignore[return-value]

    @property
    def _brush_circle_overlay(self) -> BrushCircleOverlay:
        return self._overlays['brush_circle']  # type: ignore[return-value]

    def _update_viewer_grid(self) -> None:
        """Keep viewer grid settings up to date with settings values."""

        settings = get_settings()

        self.grid.stride = settings.application.grid_stride
        self.grid.shape = (
            settings.application.grid_height,
            settings.application.grid_width,
        )
        self.grid.spacing = settings.application.grid_spacing

    @property
    def background_color(self) -> ColorValue:
        if self.background_color_override is not None:
            return self.background_color_override

        return ColorValue(
            get_theme(get_settings().appearance.theme).canvas.as_rgb_tuple()
        )

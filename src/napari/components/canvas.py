from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field

from napari.components.grid import GridCanvas
from napari.components.overlays import (
    BrushCircleOverlay,
    CanvasOverlay,
    CurrentSliceOverlay,
    FloatingAxesOverlay,
    ScaleBarOverlay,
    TextOverlay,
    ZoomOverlay,
)
from napari.settings import get_settings
from napari.utils.color import ColorValue
from napari.utils.events import Event, EventedDictNamespace, EventedModel
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from napari.components import LayerList


class Orientation(StrEnum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


class OverlayTiling(EventedModel):
    """Overlay tiling controls.

    For each canvas position, tiling direction can be set to vertical or horizontal.
    Padding between tiles can also be changed as a (vertical, horizontal) tuple.

    Examples
    --------
    >>> canvas = Canvas()
    >>> canvas.overlay_tiling.top_right = 'vertical'
    """

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
    background_color_override : ColorValue or None
        Override the theme background color with a custom one.
    grid : GridCanvas
        A model that controls the enabling and settings of the grid mode.
    overlays : EventedDictNamespace
        A dictionary/namespace containing canvas overlays. By default, it exposes
        publicly 'scale_bar', 'text', 'current_slice' and 'floating_axes'
    overlay_tiling : OverlayTiling
        Controls for the overlay tiling direction and padding.
    size : tuple[int, int]
        The canvas size following the Numpy convention of height x width
    """

    background_color_override: ColorValue | None = None
    grid: GridCanvas = Field(default_factory=GridCanvas, frozen=True)
    overlays: EventedDictNamespace[CanvasOverlay] = Field(
        default_factory=lambda: EventedDictNamespace[CanvasOverlay](
            {
                'scale_bar': ScaleBarOverlay(),
                'text': TextOverlay(),
                '_brush_circle': BrushCircleOverlay(),
                '_zoom_box': ZoomOverlay(),
                'current_slice': CurrentSliceOverlay(),
                'floating_axes': FloatingAxesOverlay(),
            }
        )
    )
    overlay_tiling: OverlayTiling = Field(
        default_factory=OverlayTiling, frozen=True
    )
    size: tuple[int, int] = (800, 600)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.events.add(_overlay_positions_changed=Event)
        self._viewer_theme = None

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

        settings.appearance.events.theme.connect(self.events.background_color)

    def viewbox_size(self, layers: LayerList) -> tuple[int, int]:
        """Get the size of a single viewbox (whether grid is enabled or not).

        If grid.border_width > 0, that's accounted for too.
        """
        viewbox_size = np.array(self.size)
        if self.grid.enabled:
            grid_shape = np.array(self.grid.actual_shape(layers))
            spacing_pixels = self.grid._compute_canvas_spacing(
                self.size, layers
            )
            # Now calculate actual available space
            total_gap_space = spacing_pixels * (grid_shape - 1)
            available_space = self.size - total_gap_space
            viewbox_size = available_space / grid_shape
        return tuple(viewbox_size)

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
            return self.background_color_override.copy()

        # viewer theme can be different than the global settings theme
        theme = (
            get_settings().appearance.theme
            if self._viewer_theme is None
            else self._viewer_theme
        )

        return ColorValue(
            np.array(get_theme(theme).canvas.as_rgb_tuple()) / 255
        )

    def _update_bgcolor_from_viewer(self, theme: Event) -> None:
        changed = theme.value != self._viewer_theme
        self._viewer_theme = theme.value
        if changed and self.background_color_override is None:
            self.events.background_color()

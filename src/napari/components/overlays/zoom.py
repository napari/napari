"""Zoom-box."""

from __future__ import annotations

from typing import Any

from pydantic import field_validator

from napari.components.overlays.base import CanvasOverlay


class ZoomOverlay(CanvasOverlay):
    """A box that can be used to select and transform objects.

    Attributes
    ----------
    position : 2-tuple of 2-tuples
        Corners at the top left and bottom right in canvas coordinates.
    zoom_area : 2-tuple of 2-tuples
        Coordinates in world space of the area to zoom in.
    box : bool
        Whether the background box is visible or not.
    box_color : ColorValue or None
        Background box color. If unset, it defaults to the canvas color.
    gridded : bool
        The overlay will be duplicated across all grid cells in gridded mode.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    blending : Blending
        One of a list of preset blending modes that determines how RGB and
        alpha values of the overlay get mixed with the visuals below.
    """

    position: tuple[tuple[float, float], tuple[float, float]] = (
        (0, 0),
        (0, 0),
    )
    zoom_area: tuple[tuple[float, float], tuple[float, float]] = (
        (0, 0),
        (0, 0),
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @field_validator('position', 'zoom_area', mode='before')
    @classmethod
    def _validate_bounds(
        cls, v: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        tup_1, tup_2 = v
        x1, y1 = (float(coord) for coord in tup_1)
        x2, y2 = (float(coord) for coord in tup_2)
        return (x1, y1), (x2, y2)

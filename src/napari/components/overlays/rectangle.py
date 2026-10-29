"""Zoom-box."""

from __future__ import annotations

from pydantic import field_validator

from napari.components.overlays.base import CanvasOverlay


class _RectOverlay(CanvasOverlay):
    """A rectangle that can be used to select object.

    Attributes
    ----------
    corners_canvas : 2-tuple of 2-tuples
        Corners at the top left and bottom right in canvas coordinates.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    position: None = None  # not used for this overlay

    corners_canvas: tuple[tuple[float, float], tuple[float, float]] = (
        (0, 0),
        (0, 0),
    )

    @field_validator('corners_canvas', mode='before')
    @classmethod
    def _validate_canvas(
        cls, v: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        tup_1, tup_2 = v
        x1, y1 = (float(coord) for coord in tup_1)
        x2, y2 = (float(coord) for coord in tup_2)
        return (x1, y1), (x2, y2)


class ZoomRectOverlay(_RectOverlay):
    pass


class SelectionRectOverlay(_RectOverlay):
    pass

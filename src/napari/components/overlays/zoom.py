"""Zoom-box."""

from __future__ import annotations

from napari._pydantic_compat import validator
from napari.components.overlays.base import CanvasOverlay
from napari.utils.events import Event


class ZoomOverlay(CanvasOverlay):
    """A box that can be used to select and transform objects.

    Attributes
    ----------
    canvas_positions : 2-tuple of 2-tuples
        Corners at the top left and bottom right in canvas coordinates.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    canvas_positions: tuple[tuple[float, float], tuple[float, float]] = (
        (0, 0),
        (0, 0),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events.add(zoom=Event)

    @validator('canvas_positions', pre=True, always=True, allow_reuse=True)
    def _validate_bounds(
        cls, v: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        tup_1, tup_2 = v
        return tuple(tup_1), tuple(tup_2)  # type: ignore[return-value]

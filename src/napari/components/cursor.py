import numpy as np

from napari.utils.events import EventedModel


class Cursor(EventedModel):
    """Cursor object with position and properties of the cursor.

    Canvas-related attributes are Read-Only and set internally by napari.

    Attributes
    ----------
    position : tuple of float
        Position of the cursor in world coordinates. If the cursor is outside of,
        the canvas, then the last known position is stored instead.
    canvas_position : tuple of int or None
        Position of the cursor in canvas pixel coordinates (y, x).
        None when cursor is outside the canvas.
    viewbox : tuple[int, int] or None
        Position of the cursor in the grid.
    """

    # fields
    position: tuple[float, ...] = (1.0, 1.0)
    _canvas_position: tuple[int, int] | None = None
    _view_direction: np.ndarray[tuple[int], np.dtype[np.floating]] | None = (
        None
    )
    _viewbox: tuple[int, int] | None = None

    @property
    def canvas_position(self) -> tuple[int, int] | None:
        return self._canvas_position

    @property
    def viewbox(self) -> tuple[int, int] | None:
        return self._viewbox

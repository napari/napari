import numpy as np

from napari.components._viewer_constants import CursorStyle
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
    view_direction : Optional[np.ndarray]
        The vector describing the direction of the camera in the scene
        in world coordinates. This is None when viewing in 2D.
    viewbox : tuple[int, int] or None
        Position of the cursor in the grid.
    scaled : bool
        Flag to indicate whether cursor size should be scaled to zoom.
        Only relevant for circle and square cursors which are drawn
        with a particular size.
    size : float
        Size of the cursor in canvas pixels. Only relevant for circle
        and square cursors which are drawn with a particular size.
    style : str
        Style of the cursor. Must be one of
            * square: A square
            * circle: A circle
            * cross: A cross
            * forbidden: A forbidden symbol
            * pointing: A finger for pointing
            * standard: The standard cursor
            # crosshair: A crosshair
    """

    # fields
    position: tuple[float, ...] = (1.0, 1.0)
    scaled: bool = True
    size: float = 1.0
    style: CursorStyle = CursorStyle.STANDARD
    _canvas_position: tuple[int, int] | None = None
    _view_direction: np.ndarray[tuple[int], np.dtype[np.floating]] | None = (
        None
    )
    _viewbox: tuple[int, int] | None = None

    @property
    def canvas_position(self) -> tuple[int, int] | None:
        return self._canvas_position

    @property
    def view_direction(
        self,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]] | None:
        return self._view_direction

    @property
    def viewbox(self) -> tuple[int, int] | None:
        return self._viewbox

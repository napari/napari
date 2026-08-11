from __future__ import annotations

from typing import TYPE_CHECKING

from napari.components._viewer_constants import CursorStyle
from napari.utils.events import EventedModel

if TYPE_CHECKING:
    import numpy as np

    from napari.components import Camera


class Cursor(EventedModel):
    """Cursor object with position and properties of the cursor.

    Attributes
    ----------
    position : tuple of float
        Position of the cursor in world coordinates. If the cursor is outside of,
        the canvas, then the last known position is stored instead.
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
    viewbox: tuple[int, int] | None = None
    scaled: bool = True
    size: float = 1.0
    style: CursorStyle = CursorStyle.STANDARD

    def view_direction(
        self,
        camera: Camera,
        canvas_size: tuple[int, int],
        canvas_position: tuple[float, float],
        ndim: int,
        dims_displayed: tuple[int, ...],
    ) -> np.ndarray | None:
        """Calculate the view direction at the cursor's canvas position.

        Parameters
        ----------
        camera : Camera
            The camera model used to calculate the view direction.
        canvas_size : tuple of int
            Size of the canvas in pixels, as ``(height, width)``.
        canvas_position : tuple of float
            Position of the cursor in the canvas in pixels, as ``(x, y)`` where
            ``x`` is the column and ``y`` is the row.
        ndim : int
            Number of dimensions of the scene.
        dims_displayed : tuple of int
            Dimensions being displayed in the viewer.

        Returns
        -------
        view_direction : np.ndarray or None
            nD view direction vector as an ``(ndim,)`` array, or ``None`` when
            fewer than three dimensions are displayed.
        """
        return camera.calculate_nd_view_direction(
            ndim=ndim,
            dims_displayed=dims_displayed,
            canvas_position=canvas_position,
            canvas_size=canvas_size,
        )

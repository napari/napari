from typing import Optional

import numpy as np

from napari.components._viewer_constants import CursorStyle
from napari.utils.events import EventedModel


class Cursor(EventedModel):
    """Cursor object with position and properties of the cursor.

    Attributes
    ----------
    position : tuple of float
        Position of the cursor in world coordinates. None if outside the
        world.
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
    _view_direction : Optional[np.ndarray]
        The vector describing the direction of the camera in the scene
        in world coordinates.
        This is None when viewing in 2D.
    """

    # fields
    position: Optional[tuple[float, ...]] = (1.0, 1.0)
    # FIXME: None value of position is not compatible with ViewerModel._on_layers_change
    scaled: bool = True
    size = 1.0
    style: CursorStyle = CursorStyle.STANDARD
    _view_direction: Optional[np.ndarray] = None

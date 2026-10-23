import numpy as np
from pydantic import Field, PrivateAttr
from typing_extensions import deprecated

from napari.utils.events import EventedModel


class Cursor(EventedModel):
    """Cursor object with position and properties of the cursor.

    Attributes
    ----------
    position : tuple of float
        Position of the cursor in world coordinates. If the cursor is outside of,
        the canvas, then the last known position is stored instead.
    viewbox : tuple[int, int]
        Position of the cursor in the grid.

    .. deprecated: 0.11.0
        Cursor is deprecated; to access cursor information, use mouse callbacks
        and retrieve this information from the received Event instead.
    """

    # fields
    position: tuple[float, ...] = Field(
        (1.0, 1.0),
        deprecated=deprecated(
            'Cursor.position is deprecated; use Event.position from a mouse callback instead.'
        ),
    )
    viewbox: tuple[int, int] = Field(
        (0, 0),
        deprecated=deprecated(
            'Cursor.position is deprecated; use Event.viewbox from a mouse callback instead.'
        ),
    )
    scaled: bool = Field(
        True,
        deprecated=deprecated(
            'Cursor.scaled is deprecated and no longer has an effect; use Labels.brush_size_is_canvas_pixels instead.'
        ),
    )
    size: float = Field(
        1.0,
        deprecated=deprecated(
            'Cursor.size is deprecated and no longer has an effect; use Labels.brush_size instead.'
        ),
    )
    style: str = Field(
        'default',
        deprecated=deprecated(
            'Cursor.style is deprecated and no longer has an effect; use Viewer.window.override_cursor_style instead'
        ),
    )
    # this is here just to support viewermodel.update_status_from_cursor during deprecation
    _view_direction: np.ndarray | None = PrivateAttr(None)

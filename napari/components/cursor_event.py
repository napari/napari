from typing import Optional, Tuple

from ..utils.events import ReadOnlyModel
from ._viewer_constants import CursorEventType


class CursorEvent(ReadOnlyModel):
    """Cursor event object that gets passed to mouse callbacks.

    Attributes
    ----------
    position : tuple
        Position of the cursor in world coordinates.
    data_position : tuple
        Position of the cursor in data coordinates if the
        cursor event has been passed to a specific layer.
        None otherwise.
    canvas_position : tuple
        Position of the cursor in canvas.
    delta : tuple or None
        Mouse movement on last event. None if outside the
        canvas.
    type : str
        Type of last cursor event. Must be one of
        * mouse_move: A mouse move event
        * mouse_press: A mouse press event
        * mouse_release: A mouse release event
        * mouse_wheel: A mouse wheel event
    inverted : bool
        Flag if cursor movement is inverted.
    is_dragging : bool
        Flag if cursor is currently dragging.
    modifiers : tuple
        Identities of any modifier keys held during cursor event.
    """

    # fields
    position: Tuple[float, ...] = (1, 1)
    data_position: Optional[Tuple[float, ...]] = None
    canvas_position: Tuple[int, int] = (0, 0)
    delta: Tuple[float, float] = (0, 0)
    modifiers: Tuple[str, ...] = ()
    type: CursorEventType = CursorEventType.MOUSE_MOVE
    inverted: bool = False
    is_dragging: bool = False

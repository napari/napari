from typing import Tuple

from ..utils.events import ReadOnlyModel
from ._viewer_constants import CursorEventType


class CursorEvent(ReadOnlyModel):
    """Cursor event object that gets passed to mouse callbacks.

    Attributes
    ----------
    position : tuple or None
        Position of the cursor in world coordinates. None if outside the
        world.
    position_canvas : tuple or None
        Position of the cursor in canvas. None if outside the
        canvas.
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
    canvas_position: Tuple[int, int] = (0, 0)
    delta: Tuple[float, float] = (0, 0)
    modifiers: Tuple[str, ...] = ()
    type: CursorEventType = CursorEventType.MOUSE_MOVE
    inverted: bool = False
    is_dragging: bool = False

    class Config:
        # Once created the event is read only and not able to be modified
        allow_mutation = False
        # whether to populate models with the value property of enums, rather
        # than the raw enum. This may be useful if you want to serialise
        # model.dict() later
        use_enum_values = True
        # whether to validate field defaults (default: False)
        validate_all = True

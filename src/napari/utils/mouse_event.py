from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from app_model.types import KeyMod
from pydantic import BaseModel, ConfigDict
from typing_extensions import deprecated

from napari.utils.key_bindings import _appmodel_to_vispy_mods

if TYPE_CHECKING:
    from vispy.util.keys import Key


class MouseEventType(StrEnum):
    mouse_press = auto()
    mouse_move = auto()
    mouse_release = auto()
    mouse_double_click = auto()
    mouse_wheel = auto()


Vector3D: TypeAlias = np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]


class MouseEvent(BaseModel):
    """MouseEvent holds information about a mouse event.

    Parameters
    ----------
    type : MouseEventType
       String indicating the event type (e.g. mouse_press, key_release)
    button : int | None
        The button that generated this event (can be None).
        Left=1, right=2, middle=3. During a mouse drag, this
        will return the button that started the drag.
    buttons : tuple[int, ...]
        Tuple of buttons held during this event.
    modifier_keys : tuple[Modifier, ...]
        Tuple of modifier keys held during this event.
    delta : tuple[float, float]
        The amount of scrolling in horizontal and vertical direction. One
        "tick" corresponds to a delta of 1.0.
    press_event : MouseEvent | None
        The press event that was generated at the start of the current drag,
        if any.
    canvas_position : tuple[int, int]
        The position of the mouse on the canvas.
    viewbox : tuple[int, int]
        The coordinates of the grid viewbox containing the mouse.
    world_position : tuple[float, ...]
        The position of the mouse projected onto the world (at the depth of
        the camera center).
    view_direction : tuple[int, int, int]
        The direction of the camera view.
    up_direction : tuple[int, int, int]
        The direction of the camera up vector.
    camera_zoom : float
        The camera zoom level.
    dims_displayed : list[int]
        The dimensions displayed in the viewer.
    dims_point : tuple[float, ...]
        The point in data coordinates that the mouse is over.
    handled : bool
        Whether the event was already marked as handled by some other
        callback.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=True,
        extra='forbid',
    )

    # napari info
    view_direction: Vector3D
    up_direction: Vector3D
    camera_zoom: float
    canvas_position: tuple[int, int]
    viewbox: tuple[int, int]
    world_position: tuple[float, ...]
    dims_displayed: list[int]
    dims_point: tuple[float, ...]
    world_to_canvas: Callable[[np.ndarray, tuple[int, int]], np.ndarray]
    canvas_to_world: Callable[[np.ndarray, tuple[int, int]], np.ndarray]

    # hardware stuff
    type: MouseEventType
    button: int | None = None
    buttons: tuple[int, ...] = ()
    modifier_keys: tuple[KeyMod, ...] = ()
    delta: tuple[float, float] | None = None
    press_event: MouseEvent | None = None

    handled: bool = False

    @property
    @deprecated(
        'event.pos is deprecated. Use event.canvas_position instead.',
        category=FutureWarning,
        stacklevel=2,
    )
    def pos(self) -> tuple[int, int]:
        return self.canvas_position[::-1]

    @property
    @deprecated(
        'event.position is deprecated. Use event.world_position instead.',
        category=FutureWarning,
        stacklevel=2,
    )
    def position(self) -> tuple[float, ...]:
        return self.world_position

    @property
    @deprecated(
        'event.modifiers is deprecated. Use event.modifier_keys instead.',
        category=FutureWarning,
        stacklevel=2,
    )
    def modifiers(self) -> list[Key]:
        return _appmodel_to_vispy_mods(self.modifier_keys)

    @property
    def is_dragging(self) -> bool:
        return self.press_event is not None

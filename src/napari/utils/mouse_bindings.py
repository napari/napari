from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, Never

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from napari._vispy.mouse_event import NapariMouseEvent
    from napari.utils._proxies import ReadOnlyWrapper


class MousemapProvider:
    """Mix-in to add mouse binding functionality.

    Attributes
    ----------
    mouse_move_callbacks : list
        Callbacks from when mouse moves with nothing pressed.
    mouse_drag_callbacks : list
        Callbacks from when mouse is pressed, dragged, and released.
    mouse_wheel_callbacks : list
        Callbacks from when mouse wheel is scrolled.
    mouse_double_click_callbacks : list
        Callbacks from when mouse wheel is scrolled.
    """

    mouse_move_callbacks: list[Callable]
    mouse_wheel_callbacks: list[Callable]
    mouse_drag_callbacks: list[Callable]
    mouse_double_click_callbacks: list[Callable]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks: list[Callable] = []
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks: list[Callable] = []
        # hold callbacks for when mouse is double clicked
        self.mouse_double_click_callbacks: list[Callable] = []
        # Hold callbacks for when mouse wheel is scrolled
        self.mouse_wheel_callbacks: list[Callable] = []

        self._persisted_mouse_event: dict[
            Generator[object, Never, object], ReadOnlyWrapper[NapariMouseEvent]
        ] = {}
        self._mouse_drag_gen: dict[Callable, Generator] = {}
        self._mouse_move_gen: dict[Callable, Generator] = {}
        self._mouse_wheel_gen: dict[Callable, Generator] = {}


class MousemapProviderPydantic(BaseModel):
    """Mix-in to add mouse binding functionality.

    Attributes
    ----------
    mouse_move_callbacks : list
        Callbacks from when mouse moves with nothing pressed.
    mouse_drag_callbacks : list
        Callbacks from when mouse is pressed, dragged, and released.
    mouse_wheel_callbacks : list
        Callbacks from when mouse wheel is scrolled.
    mouse_double_click_callbacks : list
        Callbacks from when mouse wheel is scrolled.
    """

    mouse_move_callbacks: list[Callable] = Field(default_factory=list)
    mouse_wheel_callbacks: list[Callable] = Field(default_factory=list)
    mouse_drag_callbacks: list[Callable] = Field(default_factory=list)
    mouse_double_click_callbacks: list[Callable] = Field(default_factory=list)
    _persisted_mouse_event: dict = PrivateAttr(default_factory=dict)
    _mouse_drag_gen: dict = PrivateAttr(default_factory=dict)
    _mouse_move_gen: dict = PrivateAttr(default_factory=dict)
    _mouse_wheel_gen: dict = PrivateAttr(default_factory=dict)

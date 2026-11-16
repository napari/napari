from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, Never, TypeVar

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from napari._vispy.mouse_event import NapariMouseEvent
    from napari.utils._proxies import ReadOnlyWrapper

_T = TypeVar('_T')


class CallbackList(list[_T]):
    """List container for mouse callbacks that returns the appended callable.

    Returning the appended callable allows `@callbacks.append` to be used as a
    decorator without rebinding the decorated function symbol to `None`.

    # FIXME: Replace decorator-via-append pattern with an explicit connect/disconnect API.
    """

    def append(self, item: _T) -> _T:
        super().append(item)
        return item


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
        self.mouse_move_callbacks: list[Callable] = CallbackList()
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks: list[Callable] = CallbackList()
        # hold callbacks for when mouse is double clicked
        self.mouse_double_click_callbacks: list[Callable] = CallbackList()
        # Hold callbacks for when mouse wheel is scrolled
        self.mouse_wheel_callbacks: list[Callable] = CallbackList()

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

    mouse_move_callbacks: list[Callable] = Field(default_factory=CallbackList)
    mouse_wheel_callbacks: list[Callable] = Field(default_factory=CallbackList)
    mouse_drag_callbacks: list[Callable] = Field(default_factory=CallbackList)
    mouse_double_click_callbacks: list[Callable] = Field(
        default_factory=CallbackList
    )
    _persisted_mouse_event: dict = PrivateAttr(default_factory=dict)
    _mouse_drag_gen: dict = PrivateAttr(default_factory=dict)
    _mouse_move_gen: dict = PrivateAttr(default_factory=dict)
    _mouse_wheel_gen: dict = PrivateAttr(default_factory=dict)

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari.utils.events import Event

ModeCallable: TypeAlias = Callable[
    ['Layer', 'Event'], None | Generator[None, None, None]
]


class MousemapProvider:
    """Mix-in to add mouse binding functionality.

    Callbacks can be registered to respond to mouse events such as move,
    drag, wheel, and double click.

    Callbacks should provide the signature defined by `ModeCallable`, i.e.

    def callback(layer: Layer, event: Event) -> None | Generator[None, None, None]:
        ...

    Attributes
    ----------
    mouse_move_callbacks : list[ModeCallable]
        Callbacks from when mouse moves with nothing pressed.
    mouse_drag_callbacks : list[ModeCallable]
        Callbacks from when mouse is pressed, dragged, and released.
    mouse_wheel_callbacks : list[ModeCallable]
        Callbacks from when mouse wheel is scrolled.
    mouse_double_click_callbacks : list[ModeCallable]
        Callbacks from when mouse wheel is scrolled.
    """

    mouse_move_callbacks: list[ModeCallable]
    mouse_wheel_callbacks: list[ModeCallable]
    mouse_drag_callbacks: list[ModeCallable]
    mouse_double_click_callbacks: list[ModeCallable]

    # these args are required to preserve MRO call order,
    # as this class is inherited as a mixin class
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks = []
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks = []
        # hold callbacks for when mouse is double clicked
        self.mouse_double_click_callbacks = []
        # Hold callbacks for when mouse wheel is scrolled
        self.mouse_wheel_callbacks = []

        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}
        self._mouse_wheel_gen = {}

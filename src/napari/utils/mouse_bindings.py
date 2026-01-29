from collections.abc import Callable

from pydantic import BaseModel, Field, PrivateAttr


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

    mouse_move_callbacks: list[callable]
    mouse_wheel_callbacks: list[callable]
    mouse_drag_callbacks: list[callable]
    mouse_double_click_callbacks: list[callable]

    def __init__(self, *args, **kwargs) -> None:
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
    _mouse_wheel_gen: dict = PrivateAttr(default_factory=dict)

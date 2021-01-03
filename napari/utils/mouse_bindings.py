from typing import Dict, List

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
    """

    def __init__(self):
        super().__init__()
        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks = []
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks = []
        # Hold callbacks for when mouse wheel is scrolled
        self.mouse_wheel_callbacks = []

        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}
        self._mouse_wheel_gen = {}


class MousemapProviderMixin(BaseModel):
    """Mix-in to add mouse binding functionality.

    Attributes
    ----------
    mouse_move_callbacks : list
        Callbacks from when mouse moves with nothing pressed.
    mouse_drag_callbacks : list
        Callbacks from when mouse is pressed, dragged, and released.
    mouse_wheel_callbacks : list
        Callbacks from when mouse wheel is scrolled.
    """

    # Hold callbacks for when mouse moves with nothing pressed
    mouse_move_callbacks: List = Field(default_factory=list)
    # Hold callbacks for when mouse is pressed, dragged, and released
    mouse_drag_callbacks: List = Field(default_factory=list)
    # Hold callbacks for when mouse wheel is scrolled
    mouse_wheel_callbacks: List = Field(default_factory=list)

    _persisted_mouse_event: Dict = PrivateAttr(default_factory=dict)
    _mouse_drag_gen: Dict = PrivateAttr(default_factory=dict)
    _mouse_wheel_gen: Dict = PrivateAttr(default_factory=dict)

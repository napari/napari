from typing import List


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

    mouse_move_callbacks: List[callable]
    mouse_wheel_callbacks: List[callable]
    mouse_drag_callbacks: List[callable]
    mouse_double_click_callbacks: List[callable]

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

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, GetCoreSchemaHandler, PrivateAttr
from pydantic_core import CoreSchema, core_schema


class MouseCallbackList(list[Callable]):
    """A list of mouse callbacks that can also be used as a decorator.

    ``list.append`` returns ``None``, so decorating a callback with it, as in
    ``@layer.mouse_drag_callbacks.append``, rebinds the name of the decorated
    function to ``None``, which makes it impossible to remove the callback
    again. Returning the appended callback keeps the decorated name bound to
    the function itself.
    """

    def append(self, func: Callable) -> Callable:
        """Append a callback and return it, so this can be used as a decorator."""
        super().append(func)
        return func

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.list_schema(core_schema.callable_schema()),
        )


class MousemapProvider:
    """Mix-in to add mouse binding functionality.

    Attributes
    ----------
    mouse_move_callbacks : MouseCallbackList
        Callbacks from when mouse moves with nothing pressed.
    mouse_drag_callbacks : MouseCallbackList
        Callbacks from when mouse is pressed, dragged, and released.
    mouse_wheel_callbacks : MouseCallbackList
        Callbacks from when mouse wheel is scrolled.
    mouse_double_click_callbacks : MouseCallbackList
        Callbacks from when mouse wheel is scrolled.
    """

    mouse_move_callbacks: MouseCallbackList
    mouse_wheel_callbacks: MouseCallbackList
    mouse_drag_callbacks: MouseCallbackList
    mouse_double_click_callbacks: MouseCallbackList

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Hold callbacks for when mouse moves with nothing pressed
        self.mouse_move_callbacks = MouseCallbackList()
        # Hold callbacks for when mouse is pressed, dragged, and released
        self.mouse_drag_callbacks = MouseCallbackList()
        # hold callbacks for when mouse is double clicked
        self.mouse_double_click_callbacks = MouseCallbackList()
        # Hold callbacks for when mouse wheel is scrolled
        self.mouse_wheel_callbacks = MouseCallbackList()

        self._persisted_mouse_event = {}
        self._mouse_drag_gen = {}
        self._mouse_wheel_gen = {}


class MousemapProviderPydantic(BaseModel):
    """Mix-in to add mouse binding functionality.

    Attributes
    ----------
    mouse_move_callbacks : MouseCallbackList
        Callbacks from when mouse moves with nothing pressed.
    mouse_drag_callbacks : MouseCallbackList
        Callbacks from when mouse is pressed, dragged, and released.
    mouse_wheel_callbacks : MouseCallbackList
        Callbacks from when mouse wheel is scrolled.
    mouse_double_click_callbacks : MouseCallbackList
        Callbacks from when mouse wheel is scrolled.
    """

    mouse_move_callbacks: MouseCallbackList = Field(
        default_factory=MouseCallbackList
    )
    mouse_wheel_callbacks: MouseCallbackList = Field(
        default_factory=MouseCallbackList
    )
    mouse_drag_callbacks: MouseCallbackList = Field(
        default_factory=MouseCallbackList
    )
    mouse_double_click_callbacks: MouseCallbackList = Field(
        default_factory=MouseCallbackList
    )
    _persisted_mouse_event: dict = PrivateAttr(default_factory=dict)
    _mouse_drag_gen: dict = PrivateAttr(default_factory=dict)
    _mouse_wheel_gen: dict = PrivateAttr(default_factory=dict)

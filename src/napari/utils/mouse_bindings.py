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

    # Note: We don't use type annotations here to avoid Pydantic V2 picking these
    # up as model fields. We use object.__setattr__ to bypass Pydantic's
    # validate_assignment when this mixin is used with EventedModel.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Use object.__setattr__ to bypass Pydantic's validate_assignment
        # when this mixin is combined with an EventedModel subclass.
        # Hold callbacks for when mouse moves with nothing pressed
        object.__setattr__(self, 'mouse_move_callbacks', [])
        # Hold callbacks for when mouse is pressed, dragged, and released
        object.__setattr__(self, 'mouse_drag_callbacks', [])
        # hold callbacks for when mouse is double clicked
        object.__setattr__(self, 'mouse_double_click_callbacks', [])
        # Hold callbacks for when mouse wheel is scrolled
        object.__setattr__(self, 'mouse_wheel_callbacks', [])

        object.__setattr__(self, '_persisted_mouse_event', {})
        object.__setattr__(self, '_mouse_drag_gen', {})
        object.__setattr__(self, '_mouse_wheel_gen', {})

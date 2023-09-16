"""Key combinations are represented in the form ``[modifier-]key``,
e.g. ``a``, ``Control-c``, or ``Control-Alt-Delete``.
Valid modifiers are Control, Alt, Shift, and Meta.

Letters will always be read as upper-case.
Due to the native implementation of the key system, Shift pressed in
certain key combinations may yield inconsistent or unexpected results.
Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
Control is swapped with Meta such that pressing Command reads as Control.

Special keys include Shift, Control, Alt, Meta, Up, Down, Left, Right,
PageUp, PageDown, Insert, Delete, Home, End, Escape, Backspace, F1,
F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Space, Enter, and Tab

Functions take in only one argument: the parent that the function
was bound to.

By default, all functions are assumed to work on key presses only,
but can be denoted to work on release too by separating the function
into two statements with the yield keyword::

    @viewer.bind_key('h')
    def hello_world(viewer):
        # on key press
        viewer.status = 'hello world!'

        yield

        # on key release
        viewer.status = 'goodbye world :('

To create a keymap that will block others, ``bind_key(..., ...)```.
"""


import sys
import warnings
from typing import Callable, Mapping, Union

from app_model.expressions import parse_expression
from app_model.types import Action, KeyBinding, KeyBindingRule, KeyCode

from napari.utils.key_bindings.constants import KeyBindingWeights
from napari.utils.translations import trans

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    EllipsisType = type(Ellipsis)

KeyBindingLike = Union[KeyBinding, str, int]
Keymap = Mapping[
    Union[KeyBinding, EllipsisType], Union[Callable, EllipsisType]
]

KEY_SUBS = {
    'Control': 'Ctrl',
    'Option': 'Alt',
}

_UNDEFINED = object()


def coerce_keybinding(key_bind: KeyBindingLike) -> KeyBinding:
    """Convert a keybinding-like object to a KeyBinding.

    Parameters
    ----------
    key_bind : keybinding-like
        Object to coerce.

    Returns
    -------
    key_bind : KeyBinding
        Object as KeyBinding.
    """
    if isinstance(key_bind, str):
        for k, v in KEY_SUBS.items():
            key_bind = key_bind.replace(k, v)

    key_bind = KeyBinding.validate(key_bind)

    # remove redundant modifiers e.g. Shift+Shift
    for part in key_bind.parts:
        if part.key == KeyCode.Ctrl:
            part.ctrl = False
        elif part.key == KeyCode.Shift:
            part.shift = False
        elif part.key == KeyCode.Alt:
            part.alt = False
        elif part.key == KeyCode.Meta:
            part.meta = False

    return key_bind


def bind_key(
    keymap: Keymap,
    key_bind: Union[KeyBindingLike, EllipsisType],
    func=_UNDEFINED,
    *,
    overwrite=False,
):
    """This function is deprecated and will be removed in a future version.

    Bind a key combination to a keymap.

    Parameters
    ----------
    keymap : dict of str: callable
        Keymap to modify.
    key_bind : keybinding-like or ...
        Key combination.
        ``...`` acts as a wildcard if no key combinations can be matched
        in the keymap (this will overwrite all key combinations
        further down the lookup chain).
    func : callable, None, or ...
        Callable to bind to the key combination.
        If ``None`` is passed, unbind instead.
        ``...`` acts as a blocker, effectively unbinding the key
        combination for all keymaps further down the lookup chain.
    overwrite : bool, keyword-only, optional
        Whether to overwrite the key combination if it already exists.

    Returns
    -------
    unbound : callable or None
        Callable unbound by this operation, if any.

    Notes
    -----
    Key combinations are represented in the form ``[modifier-]key``,
    e.g. ``a``, ``Control-c``, or ``Control-Alt-Delete``.
    Valid modifiers are Control, Alt, Shift, and Meta.

    Letters will always be read as upper-case.
    Due to the native implementation of the key system, Shift pressed in
    certain key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
    Control is swapped with Meta such that pressing Command reads as Control.

    Special keys include Shift, Control, Alt, Meta, Up, Down, Left, Right,
    PageUp, PageDown, Insert, Delete, Home, End, Escape, Backspace, F1,
    F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Space, Enter, and Tab

    Functions take in only one argument: the parent that the function
    was bound to.

    By default, all functions are assumed to work on key presses only,
    but can be denoted to work on release too by separating the function
    into two statements with the yield keyword::

        @viewer.bind_key('h')
        def hello_world(viewer):
            # on key press
            viewer.status = 'hello world!'

            yield

            # on key release
            viewer.status = 'goodbye world :('

    To create a keymap that will block others, ``bind_key(..., ...)```.
    """
    warnings.warn(
        trans._(
            'This function is deprecated and will be removed in version 0.6.0'
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    return func


class KeymapProvider:
    """Deprecated and will be removed in version 0.6.0.

    Mix-in to add keymap functionality.

    Attributes
    ----------
    class_keymap : dict
        Class keymap.
    keymap : dict
        Instance keymap.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keymap = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if 'class_keymap' not in cls.__dict__:
            # if in __dict__, was defined in class and not inherited
            cls.class_keymap = {}
        else:
            cls.class_keymap = {
                coerce_keybinding(k): v for k, v in cls.class_keymap.items()
            }

    @classmethod
    def bind_key(
        cls,
        key_bind: Union[KeyBindingLike, EllipsisType],
        func=_UNDEFINED,
        *,
        overwrite=False,
    ):
        warnings.warn(
            trans._(
                'This function is deprecated and will be removed in version 0.6.0. Shortcuts set via the GUI will overwrite this.'
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        if key_bind is Ellipsis:
            raise TypeError(
                'Removed functionality: cannot use ellipsis as key binding.'
            )

        from napari._app_model._app import get_app

        app = get_app()

        kb = coerce_keybinding(key_bind)

        try:
            type_string = cls._type_string()
            when = parse_expression(
                f"num_selected_layers == 1 and active_layer_type == '{type_string}'"
            )
        except AssertionError:
            when = None

        if func is None:
            app.keybindings.register_keybinding_rule(
                '',
                KeyBindingRule(
                    primary=kb,
                    when=when,
                    weight=KeyBindingWeights.USER,
                ),
            )

        def inner(_func: Callable) -> Callable:
            name = f'autogen:{_func.__qualname__}'
            print(name)

            app.register_action(
                Action(
                    id=name,
                    callback=_func,
                    title=_func.__name__,
                    keybindings=[
                        KeyBindingRule(
                            primary=kb,
                            when=when,
                            weight=KeyBindingWeights.USER,
                        )
                    ],
                )
            )

            return _func

        if func is _UNDEFINED:
            return inner

        return inner(func)

"""Key combinations are represented in the form ``[modifier-]key``,
e.g. ``a``, ``Control-c``, or ``Control-Alt-Delete``.
Valid modifiers are Control, Alt, Shift, and Meta.

Letters will always be read as upper-case.
Due to the native implementation of the key system, Shift pressed in certain
key combinations may yield inconsistent or unexpected results.
Therefore, it is not recommended to use Shift with non-letter keys.
On OSX, Control is swapped with Meta such that pressing Command reads as
Control.

Special keys include Shift, Control, Alt, Meta, Up, Down, Left, Right,
PageUp, PageDown, Insert, Delete, Home, End, Escape, Backspace, F1,
F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Space, Enter, and Tab

Functions take in only one argument: the parent that the function
was bound to. This is the viewer or layer.

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

"""

import re
import types
from collections import OrderedDict, UserDict

from vispy.util import keys


SPECIAL_KEYS = [
    keys.SHIFT,
    keys.CONTROL,
    keys.ALT,
    keys.META,
    keys.UP,
    keys.DOWN,
    keys.LEFT,
    keys.RIGHT,
    keys.PAGEUP,
    keys.PAGEDOWN,
    keys.INSERT,
    keys.DELETE,
    keys.HOME,
    keys.END,
    keys.ESCAPE,
    keys.BACKSPACE,
    keys.F1,
    keys.F2,
    keys.F3,
    keys.F4,
    keys.F5,
    keys.F6,
    keys.F7,
    keys.F8,
    keys.F9,
    keys.F10,
    keys.F11,
    keys.F12,
    keys.SPACE,
    keys.ENTER,
    keys.TAB,
]

MODIFIER_KEYS = [keys.CONTROL, keys.ALT, keys.SHIFT, keys.META]


def parse_key_combo(key_combo):
    """Parse a key combination into its components in a comparable format.

    Parameters
    ----------
    key_combo : str
        Key combination.

    Returns
    -------
    key : str
        Base key of the combination.
    modifiers : set of str
        Modifier keys of the combination.
    """
    parsed = re.split('-(?=.+)', key_combo)
    *modifiers, key = parsed

    return key, set(modifiers)


def components_to_key_combo(key, modifiers):
    """Combine components to become a key combination.

    Modifier keys will always be combined in the same order:
    Control, Alt, Shift, Meta

    Letters will always be read as upper-case.
    Due to the native implementation of the key system, Shift pressed in certain
    key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys.
    On OSX, Control is swapped with Meta such that pressing Command reads as
    Control.

    Parameters
    ----------
    key : str or vispy.app.Key
        Base key.
    modifiers : combination of str or vispy.app.Key
        Modifier keys.

    Returns
    -------
    key_combo : str
        Generated key combination.
    """
    if len(key) == 1 and key.isalpha():  # it's a letter
        key = key.upper()
        cond = lambda m: True  # noqa: E731
    elif key in SPECIAL_KEYS:
        # remove redundant information i.e. an output of 'Shift-Shift'
        cond = lambda m: m != key  # noqa: E731
    else:
        # Shift is consumed to transform key

        # bug found on OSX: Command will cause Shift to not
        # transform the key so do not consume it
        # note: 'Control' is OSX Command key
        cond = lambda m: m != 'Shift' or 'Control' in modifiers  # noqa: E731

    modifiers = tuple(
        key.name
        for key in filter(
            lambda key: key in modifiers and cond(key), MODIFIER_KEYS
        )
    )

    return '-'.join(modifiers + (key,))


def normalize_key_combo(key_combo):
    """Normalize key combination to make it easily comparable.

    All aliases are converted and modifier orders are fixed to:
    Control, Alt, Shift, Meta

    Letters will always be read as upper-case.
    Due to the native implementation of the key system, Shift pressed in certain
    key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys.
    On OSX, Control is swapped with Meta such that pressing Command reads as
    Control.

    Parameters
    ----------
    key_combo : str
        Key combination.

    Returns
    -------
    normalized_key_combo : str
        Normalized key combination.
    """
    key, modifiers = parse_key_combo(key_combo)

    if len(key) != 1 and key not in SPECIAL_KEYS:
        raise TypeError(f'invalid key {key}')

    for modifier in modifiers:
        if modifier not in MODIFIER_KEYS:
            raise TypeError(f'invalid modifier key {modifier}')

    return components_to_key_combo(key, modifiers)


UNDEFINED = object()


def bind_key(keymap, key, func=UNDEFINED, *, overwrite=False):
    """Bind a key combination to a keymap.

    Parameters
    ----------
    keymap : dict of str: callable
        Keymap to modify.
    key : str
        Key combination.
    func : callable or None
        Callable to bind to the key combination.
        If ``None`` is passed, unbind instead.
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
    Due to the native implementation of the key system, Shift pressed in certain
    key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys.
    On OSX, Control is swapped with Meta such that pressing Command reads as
    Control.

    Special keys include Shift, Control, Alt, Meta, Up, Down, Left, Right,
    PageUp, PageDown, Insert, Delete, Home, End, Escape, Backspace, F1,
    F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, Space, Enter, and Tab

    Functions take in only one argument: the parent that the function
    was bound to. This is the viewer or layer.

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

    """
    if func is UNDEFINED:

        def inner(func):
            bind_key(keymap, key, func, overwrite=overwrite)
            return func

        return inner

    key = normalize_key_combo(key)

    if func is not None and key in keymap and not overwrite:
        raise ValueError(
            f'key combination {key} already used! '
            "specify 'overwrite=True' to bypass this check"
        )

    unbound = keymap.pop(key, None)

    if func is not None:
        if not callable(func):
            raise TypeError("'func' must be a callable")
        keymap[key] = func

    return unbound


class KeybindingDescriptor:
    """Descriptor which transforms ``func`` into a method with the first argument bound
    to ``class_keymap`` or ``_keymap`` depending on if it was called
    from the class or the instance, respectively.

    Parameters
    ----------
    func : callable
        Function to bind.
    """

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, klass):
        if instance is None:  # used on class
            try:
                keymap = klass.class_keymap
            except AttributeError:
                return self.__func__
        else:
            keymap = instance._keymap

        return types.MethodType(self.__func__, keymap)


class InheritedKeymap(UserDict):
    """Dictionary which inherits from another.

    Values of ``None`` are treated as though the key doesn't exist.

    Parameters
    ----------
    parent : callable() -> dict
        Parent keymap.
    contents : dict, optional
        Contents with which to initialize.
    """

    def __init__(self, parent, contents={}):
        super().__init__(contents)
        self._parent = parent

    def __getitem__(self, key):
        try:
            val = super().__getitem__(key)
        except KeyError:
            val = self._parent()[key]

        if val is None:
            raise KeyError(key)

        return val

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __delitem__(self, key):
        self[key] = None

    def copy(self):
        return InheritedKeymap(self._parent, self.data.copy())

    def pop(self, key, default=UNDEFINED):
        try:
            default = self[key]
        except KeyError:
            if default is UNDEFINED:
                raise
        else:
            del self[key]

        return default


class KeymapMixin:
    """Mix-in to add keymap functionality. Must still define ``class_keymap``
    in every subclass.
    """

    def __init__(self):
        self.keymap = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if 'class_keymap' not in cls.__dict__:
            # if in __dict__, was defined in class and not inherited
            cls.class_keymap = {}

    @property
    def keymap(self):
        """InheritedKeymap : Keymap used for shortcuts.
        Inherits from ``class_keymap``.

        Do not directly set key bindings; use ``bind_key`` instead.
        """
        return self._keymap.copy()

    @keymap.setter
    def keymap(self, keymap):
        self._keymap = InheritedKeymap(lambda: self.class_keymap, keymap)

    bind_key = KeybindingDescriptor(bind_key)

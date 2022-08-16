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


import contextlib
import inspect
import re
import time
import types
import typing
from collections import ChainMap

from vispy.util import keys

from ..settings import get_settings
from ..utils.translations import trans

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

KEY_SUBS = {'Ctrl': 'Control'}

USER_KEYMAP = {}


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
    Due to the native implementation of the key system, Shift pressed in
    certain key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
    Control is swapped with Meta such that pressing Command reads as Control.

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
    Due to the native implementation of the key system, Shift pressed in
    certain key combinations may yield inconsistent or unexpected results.
    Therefore, it is not recommended to use Shift with non-letter keys. On OSX,
    Control is swapped with Meta such that pressing Command reads as Control.

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
        raise TypeError(
            trans._(
                'invalid key {key}',
                deferred=True,
                key=key,
            )
        )

    for modifier in modifiers:
        if modifier in KEY_SUBS.keys():
            modifiers.remove(modifier)
            modifier = KEY_SUBS[modifier]

            modifiers.add(modifier)
        if modifier not in MODIFIER_KEYS:
            raise TypeError(
                trans._(
                    'invalid modifier key {modifier}',
                    deferred=True,
                    modifier=modifier,
                )
            )

    return components_to_key_combo(key, modifiers)


UNDEFINED = object()


def bind_key(keymap, key, func=UNDEFINED, *, overwrite=False):
    """Bind a key combination to a keymap.

    Parameters
    ----------
    keymap : dict of str: callable
        Keymap to modify.
    key : str or ...
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
    if func is UNDEFINED:

        def inner(func):
            bind_key(keymap, key, func, overwrite=overwrite)
            return func

        return inner

    if key is not Ellipsis:
        key = normalize_key_combo(key)

    if func is not None and key in keymap and not overwrite:
        raise ValueError(
            trans._(
                'key combination {key} already used! specify \'overwrite=True\' to bypass this check',
                deferred=True,
                key=key,
            )
        )

    unbound = keymap.pop(key, None)

    if func is not None:
        if func is not Ellipsis and not callable(func):
            raise TypeError(
                trans._(
                    "'func' must be a callable",
                    deferred=True,
                )
            )
        keymap[key] = func

    return unbound


def get_user_keymap():
    """Retrieve the current user keymap. The user keymap is global and takes precedent over all other keymaps.

    Returns
    -------
    user_keymap : dict of str: callable
        User keymap.
    """
    return USER_KEYMAP


def bind_user_key(key, func=UNDEFINED, *, overwrite=False):
    """Bind a key combination to the user keymap.

    Parameters
    ----------
    key : str or ...
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

        @bind_user_key('Space')
        def hello_world():
            # on key press
            print('hello world!')

            yield

            # on key release
            print('goodbye world :(')

    To create a keymap that will block others, ``bind_user_key(..., ...)```.
    """
    keymap = get_user_keymap()

    if func is UNDEFINED:

        def inner(func):
            bind_key(keymap, key, func, overwrite=overwrite)
            return func

        return inner

    if key is not Ellipsis:
        key = normalize_key_combo(key)

    if func is not None and key in keymap and not overwrite:
        raise ValueError(
            trans._(
                'key combination {key} already used! specify \'overwrite=True\' to bypass this check',
                deferred=True,
                key=key,
            )
        )

    unbound = keymap.pop(key, None)

    if func is not None:
        if func is not Ellipsis and not callable(func):
            raise TypeError(
                trans._(
                    "'func' must be a callable",
                    deferred=True,
                )
            )
        keymap[key] = func

    return unbound


class KeybindingDescriptor:
    """Descriptor which transforms ``func`` into a method with the first
    argument bound to ``class_keymap`` or ``keymap`` depending on if it was
    called from the class or the instance, respectively.

    Parameters
    ----------
    func : callable
        Function to bind.
    """

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, cls):
        keymap = instance.keymap if instance is not None else cls.class_keymap
        return types.MethodType(self.__func__, keymap)


class KeymapProvider:
    """Mix-in to add keymap functionality.

    Attributes
    ----------
    class_keymap : dict
        Class keymap.
    keymap : dict
        Instance keymap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keymap = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if 'class_keymap' not in cls.__dict__:
            # if in __dict__, was defined in class and not inherited
            cls.class_keymap = {}

    bind_key = KeybindingDescriptor(bind_key)


def _bind_keymap(keymap, instance):
    """Bind all functions in a keymap to an instance.

    Parameters
    ----------
    keymap : dict
        Keymap to bind.
    instance : object
        Instance to bind to.

    Returns
    -------
    bound_keymap : dict
        Keymap with functions bound to the instance.
    """
    bound_keymap = {
        key: types.MethodType(func, instance) if func is not Ellipsis else func
        for key, func in keymap.items()
    }
    return bound_keymap


class KeymapHandler:
    """Handle key mapping and calling functionality.

    Attributes
    ----------
    keymap_providers : list of KeymapProvider
        Classes that provide the keymaps for this class to handle.
    """

    def __init__(self):
        super().__init__()
        self._key_release_generators = {}
        self.keymap_providers = []

    @property
    def keymap_chain(self):
        """collections.ChainMap: Chain of keymaps from keymap providers."""
        maps = [get_user_keymap()]

        for parent in self.keymap_providers:
            maps.append(_bind_keymap(parent.keymap, parent))
            # For parent and superclasses add inherited keybindings
            for cls in parent.__class__.__mro__:
                if hasattr(cls, 'class_keymap'):
                    maps.append(_bind_keymap(cls.class_keymap, parent))

        return ChainMap(*maps)

    @property
    def active_keymap(self):
        """dict: Active keymap, created by resolving the keymap chain."""
        active_keymap = self.keymap_chain
        keymaps = active_keymap.maps

        for i, keymap in enumerate(keymaps):
            if Ellipsis in keymap:  # catch-all key
                # trim all keymaps after catch-all
                active_keymap = ChainMap(*keymaps[: i + 1])
                break

        active_keymap_final = {
            k: func
            for k, func in active_keymap.items()
            if func is not Ellipsis
        }

        return active_keymap_final

    def press_key(self, key_combo):
        """Simulate a key press to activate a keybinding.

        Parameters
        ----------
        key_combo : str
            Key combination.
        """
        key_combo = normalize_key_combo(key_combo)
        keymap = self.active_keymap
        if key_combo in keymap:
            func = keymap[key_combo]
        elif Ellipsis in keymap:  # catch-all
            func = keymap[...]
        else:
            return  # no keybinding found

        if func is Ellipsis:  # blocker
            return
        elif not callable(func):
            raise TypeError(
                trans._(
                    "expected {func} to be callable",
                    deferred=True,
                    func=func,
                )
            )

        generator_or_callback = func()

        if inspect.isgeneratorfunction(func):
            try:
                next(generator_or_callback)  # call function
            except StopIteration:  # only one statement
                pass
            else:
                key, _ = parse_key_combo(key_combo)
                self._key_release_generators[key] = generator_or_callback
        if isinstance(generator_or_callback, typing.Callable):
            key, _ = parse_key_combo(key_combo)
            self._key_release_generators[key] = (
                generator_or_callback,
                time.time(),
            )

    def release_key(self, key_combo):
        """Simulate a key release for a keybinding.

        Parameters
        ----------
        key_combo : str
            Key combination.
        """
        key, _ = parse_key_combo(key_combo)
        with contextlib.suppress(KeyError, StopIteration):
            val = self._key_release_generators[key]
            # val could be callback function with time to check
            # if it should be called or generator that need to make
            # additional step on key release
            if isinstance(val, tuple):
                callback, start = val
                if (
                    time.time() - start
                    > get_settings().application.hold_button_delay
                ):
                    callback()
            else:
                next(val)  # call function

    def on_key_press(self, event):
        """Callback that whenever key pressed in canvas.

        Parameters
        ----------
        event : vispy.util.event.Event
            The vispy key press event that triggered this method.
        """
        if (
            event.native is not None
            and event.native.isAutoRepeat()
            and event.key.name not in ['Up', 'Down', 'Left', 'Right']
        ) or event.key is None:
            # pass if no key is present or if key is held down, unless the
            # key being held down is one of the navigation keys
            # this helps for scrolling, etc.
            return

        combo = components_to_key_combo(event.key.name, event.modifiers)
        self.press_key(combo)

    def on_key_release(self, event):
        """Called whenever key released in canvas.

        Parameters
        ----------
        event : vispy.util.event.Event
            The vispy key release event that triggered this method.
        """
        if event.key is None or (
            # on linux press down is treated as multiple press and release
            event.native is not None
            and event.native.isAutoRepeat()
        ):
            return
        combo = components_to_key_combo(event.key.name, event.modifiers)
        self.release_key(combo)

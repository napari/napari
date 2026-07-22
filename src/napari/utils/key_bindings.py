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

from __future__ import annotations

import inspect
import time
from collections import ChainMap
from collections.abc import Callable, MutableMapping
from types import EllipsisType, MethodType
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, overload

from app_model.types import KeyBinding, KeyCode, KeyMod
from vispy.util import keys

from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Generator

    from vispy.util.event import Event

KeyBindingLike = KeyBinding | str | int
Keymap = MutableMapping[KeyBinding | EllipsisType, Callable | EllipsisType]

KeymapFunction = Callable[..., Any]

_F = TypeVar('_F', bound=KeymapFunction)

# global user keymap; to be made public later in refactoring process
USER_KEYMAP: Keymap = {}

KEY_SUBS: dict[str, str] = {
    'Super': 'Meta',
    'Command': 'Meta',
    'Cmd': 'Meta',
    'Control': 'Ctrl',
    'Option': 'Alt',
}


class _Undefined:
    """Sentinel for undefined values."""


_UNDEFINED = _Undefined()

_VISPY_SPECIAL_KEYS: list[keys.Key] = [
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

_VISPY_MODS: dict[keys.Key, KeyMod] = {
    keys.CONTROL: KeyMod.CtrlCmd,
    keys.SHIFT: KeyMod.Shift,
    keys.ALT: KeyMod.Alt,
    keys.META: KeyMod.WinCtrl,
}

# TODO: add this to app-model instead
KeyBinding.__hash__ = lambda self: hash(str(self))


def _coerce_keymap(keymap: Keymap) -> Keymap:
    """Coerce every key of a keymap to a KeyBinding."""
    return {
        k if k is Ellipsis else coerce_keybinding(k): v
        for k, v in keymap.items()
    }


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

    return KeyBinding.validate(key_bind)


@overload
def bind_key(
    keymap: Keymap,
    key_bind: KeyBindingLike | EllipsisType,
    func: _Undefined = ...,
    *,
    overwrite: bool = ...,
) -> Callable[[_F], _F]: ...


@overload
def bind_key(
    keymap: Keymap,
    key_bind: KeyBindingLike | EllipsisType,
    func: KeymapFunction | None | EllipsisType,
    *,
    overwrite: bool = ...,
) -> KeymapFunction | EllipsisType | None: ...


def bind_key(
    keymap: Keymap,
    key_bind: KeyBindingLike | EllipsisType,
    func: Callable | None | EllipsisType | _Undefined = _UNDEFINED,
    *,
    overwrite: bool = False,
) -> Callable[[_F], _F] | KeymapFunction | EllipsisType | None:
    """Bind a key combination to a keymap.

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
    Callable | None
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
    if func is _UNDEFINED:

        def inner(func: _F) -> _F:
            bind_key(keymap, key_bind, func, overwrite=overwrite)
            return func

        return inner

    key: KeyBinding | EllipsisType = (
        coerce_keybinding(key_bind) if key_bind is not Ellipsis else key_bind
    )
    if key_bind is not Ellipsis:
        key_bind = coerce_keybinding(key_bind)

    if func is not None and key in keymap and not overwrite:
        raise ValueError(
            trans._(
                "keybinding {key} already used! specify 'overwrite=True' to bypass this check",
                deferred=True,
                key=str(key),
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


def _get_user_keymap() -> Keymap:
    """Retrieve the current user keymap. The user keymap is global and takes precedent over all other keymaps.

    Returns
    -------
    user_keymap : dict of str: callable
        User keymap.
    """
    return USER_KEYMAP


@overload
def _bind_user_key(
    key_bind: KeyBindingLike,
    func: _Undefined = ...,
    *,
    overwrite: bool = ...,
) -> Callable[[_F], _F]: ...


@overload
def _bind_user_key(
    key_bind: KeyBindingLike,
    func: KeymapFunction | None | EllipsisType,
    *,
    overwrite: bool = ...,
) -> KeymapFunction | EllipsisType | None: ...


def _bind_user_key(
    key_bind: KeyBindingLike,
    func: KeymapFunction | None | EllipsisType | _Undefined = _UNDEFINED,
    *,
    overwrite: bool = False,
) -> Callable[[_F], _F] | KeymapFunction | EllipsisType | None:
    """Bind a key combination to the user keymap.

    See ``bind_key`` docs for details.
    """
    if func is _UNDEFINED:
        return bind_key(_get_user_keymap(), key_bind, overwrite=overwrite)
    return bind_key(_get_user_keymap(), key_bind, func, overwrite=overwrite)


def _vispy2appmodel(event: Event) -> KeyBinding:
    key: str = event.key.name
    modifiers: list[keys.Key] = event.modifiers
    cond: Callable[[keys.Key], bool]

    if len(key) == 1 and key.isalpha():  # it's a letter
        key = key.upper()
        cond = lambda m: True  # noqa: E731
    elif key in _VISPY_SPECIAL_KEYS:
        # remove redundant information i.e. an output of 'Shift-Shift'
        cond = lambda m: m != key  # noqa: E731
    else:
        # Shift is consumed to transform key

        # bug found on OSX: Command will cause Shift to not
        # transform the key so do not consume it
        # note: 'Control' is OSX Command key
        cond = lambda m: m != 'Shift' or 'Control' in modifiers  # noqa: E731

    kb: int = KeyCode.from_string(KEY_SUBS.get(key, key))

    for key in filter(lambda key: key in modifiers and cond(key), _VISPY_MODS):
        kb |= _VISPY_MODS[key]

    return coerce_keybinding(kb)


class KeybindingDescriptor:
    """Descriptor which transforms ``func`` into a method with the first
    argument bound to ``class_keymap`` or ``keymap`` depending on if it was
    called from the class or the instance, respectively.

    Parameters
    ----------
    func : callable
        Function to bind.
    """

    def __init__(self, func: KeymapFunction) -> None:
        self.__func__ = func

    def __get__(
        self, instance: KeymapProvider | None, cls: type[KeymapProvider]
    ) -> MethodType:
        keymap = instance.keymap if instance is not None else cls.class_keymap
        return MethodType(self.__func__, keymap)


class KeymapProvider:
    """Mix-in to add keymap functionality.

    Attributes
    ----------
    class_keymap : dict
        Class keymap.
    keymap : dict
        Instance keymap.
    """

    class_keymap: ClassVar[Keymap]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keymap: Keymap = {}

    @property
    def keymap(self) -> Keymap:
        return self._keymap

    @keymap.setter
    def keymap(self, value: Keymap) -> None:
        self._keymap = _coerce_keymap(value)

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        if 'class_keymap' not in cls.__dict__:
            # if in __dict__, was defined in class and not inherited
            cls.class_keymap = {}
        else:
            cls.class_keymap = _coerce_keymap(cls.class_keymap)

    bind_key = KeybindingDescriptor(bind_key)


def _bind_keymap(keymap: Keymap, instance: Any) -> Keymap:
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
    bound_keymap: Keymap = {
        key: MethodType(func, instance) if func is not Ellipsis else func
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

    def __init__(self) -> None:
        self._key_release_generators: dict[
            str,
            Generator[None, None, None] | tuple[Callable[[], Any], float],
        ] = {}
        self.keymap_providers: list[KeymapProvider] = []

    @property
    def keymap_chain(self) -> ChainMap[KeyBinding | EllipsisType, Any]:
        """collections.ChainMap: Chain of keymaps from keymap providers."""
        maps = [_get_user_keymap()]

        for parent in self.keymap_providers:
            maps.append(_bind_keymap(parent.keymap, parent))
            # For parent and superclasses add inherited keybindings
            for cls in parent.__class__.__mro__:
                if hasattr(cls, 'class_keymap'):
                    maps.append(_bind_keymap(cls.class_keymap, parent))

        return ChainMap(*maps)

    @property
    def active_keymap(self) -> dict[KeyBinding | EllipsisType, KeymapFunction]:
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

    def press_key(self, key_bind: KeyBindingLike) -> bool:
        """Simulate a key press to activate a keybinding.

        Parameters
        ----------
        key_bind : keybinding-like
            Key combination.
        """
        key_bind = coerce_keybinding(key_bind)
        keymap = self.active_keymap
        if key_bind in keymap:
            func = keymap[key_bind]
        elif Ellipsis in keymap:  # catch-all
            func = keymap[...]
        else:
            return False

        if func is Ellipsis:  # blocker
            return False
        if not callable(func):
            raise TypeError(
                trans._(
                    'expected {func} to be callable',
                    deferred=True,
                    func=func,
                )
            )

        generator_or_callback = func()

        key = str(key_bind.parts[-1].key)

        if inspect.isgeneratorfunction(func):
            try:
                next(generator_or_callback)  # call function
            except StopIteration:  # only one statement
                pass
            else:
                self._key_release_generators[key] = generator_or_callback
        if callable(generator_or_callback):
            self._key_release_generators[key] = (
                generator_or_callback,
                time.time(),
            )
        return True

    def release_key(self, key_bind: KeyBindingLike) -> bool:
        """Simulate a key release for a keybinding.

        Parameters
        ----------
        key_bind : keybinding-like
            Key combination.
        """
        from napari.settings import get_settings

        key_bind = coerce_keybinding(key_bind)
        key = str(key_bind.parts[-1].key)
        try:
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
        except KeyError:
            return False
        except StopIteration:
            pass
        return True

    def on_key_press(self, event: Event) -> None:
        """Called whenever key pressed in canvas.

        Parameters
        ----------
        event : vispy.util.event.Event
            The vispy key press event that triggered this method.
        """
        from napari.utils.action_manager import action_manager

        if event.key is None:
            # TODO determine when None key could be sent.
            return

        kb = _vispy2appmodel(event)

        repeatables = {
            *action_manager._get_repeatable_shortcuts(self.keymap_chain),
            'Up',
            'Down',
            'Left',
            'Right',
        }

        if (
            event.native is not None
            and event.native.isAutoRepeat()
            and kb not in repeatables
        ) or event.key is None:
            # pass if no key is present or if the shortcut combo is held down,
            # unless the combo being held down is one of the autorepeatables or
            # one of the navigation keys (helps with scrolling).
            return

        event.handled = self.press_key(kb)

    def on_key_release(self, event: Event) -> None:
        """Called whenever key released in canvas.

        Parameters
        ----------
        event : vispy.util.event.Event
            The vispy key release event that triggered this method.
        """
        if event.key is None or (
            # on linux press down is treated as multiple press and release
            event.native is not None and event.native.isAutoRepeat()
        ):
            return
        kb = _vispy2appmodel(event)
        event.handled = self.release_key(kb)

from __future__ import annotations

import inspect
import sys
import traceback
import warnings
from typing import TYPE_CHECKING, Literal

from napari.utils.key_bindings import (
    KeyBindingLike,
    KeyCode,
    coerce_keybinding,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator

    from napari._vispy.mouse_event import NapariMouseEvent
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.mouse_bindings import MousemapProvider


def _run_callbacks_and_maybe_store_generators(
    obj: MousemapProvider,
    event: ReadOnlyWrapper[NapariMouseEvent],
    callback_type: Literal['move', 'drag', 'wheel'],
    ignore: Collection[Callable] = (),
) -> None:
    """Run callbacks for the first time and store generators if any.

    When exectued on a MousemapProvider, its callbacks of the given callback_type will
    be executed. If any of these callbacks return a generator, this generator is
    stored on the object itself inside the relative persistent generator dictionary
    for later yields (e.g: various steps of a mouse_move).

    The event object is similarly stored for later use in its wrapped form (see
    _vispy/canvas.py for how NapariMouseEvent is handled). The wrapped object is
    then sneakily replaced by _step_active_generators with the new event, so the
    next yield operates on the new event values.
    """
    callbacks = getattr(obj, f'mouse_{callback_type}_callbacks')
    gen_dict = getattr(obj, f'_mouse_{callback_type}_gen')
    for func in callbacks:
        if func in gen_dict or func in ignore:
            # we're already handling this callback via generator, or we just finised handling
            # it within the same event; do not start anew
            continue

        # execute function to run it if it is a simple function, or get the generator
        gen = func(obj, event)
        if inspect.isgenerator(gen):
            # if function returns a generator then try to iterate it (the first step should
            # set up the initial state) and set up for later iterations by storing the
            # generator itself and the event wrapper
            try:
                next(gen)
            except StopIteration:
                pass
            else:
                # The event passed to the generator (and stored here) is actually a wrapper.
                # On later calls, we just replace the inner wrapped event transparently,
                # so subsequent calls of next(gen) will use the updated event values
                gen_dict[func] = gen
                obj._persisted_mouse_event[gen] = event


def _forget_generator(
    obj: MousemapProvider, gen_dict: dict, func: Callable, gen: Generator
) -> None:
    del gen_dict[func]
    del obj._persisted_mouse_event[gen]


def _warn_callback_failed(func: Callable) -> None:
    name = getattr(func, '__qualname__', repr(func))
    warnings.warn(
        f'mouse callback {name} failed with the following error:\n'
        f'{traceback.format_exc()}',
        stacklevel=2,
    )


def _step_active_generators(
    obj: MousemapProvider,
    event: ReadOnlyWrapper[NapariMouseEvent],
    callback_type: Literal['move', 'drag', 'wheel'],
) -> list[Callable]:
    """Step all active generators of type callback_type on the given MousemapProvider.

    Replaces the wrapped event with the new value, so the next yield operates on the new
    event, then calls next() on the generator.

    Returns a list of completed generators to be used by the caller to avoid recreating
    a just-completed generator from the same event.
    """
    gen_dict = getattr(obj, f'_mouse_{callback_type}_gen')
    completed = []
    for func, gen in tuple(gen_dict.items()):
        # update the wrapper with the current event
        # (see _run_callbacks_and_maybe_store_generators for an explanation)
        # plain events (e.g. from plugin tests) are used as they are
        obj._persisted_mouse_event[gen].__wrapped__ = getattr(
            event, '__wrapped__', event
        )
        try:
            next(gen)
        except StopIteration:
            # done, delete the generator and stored event
            _forget_generator(obj, gen_dict, func, gen)
            # we communicate back to the caller which generators were just completed
            completed.append(func)
        except Exception:  # noqa: BLE001
            # forget the dead generator so it cannot block later events
            _forget_generator(obj, gen_dict, func, gen)
            completed.append(func)
            _warn_callback_failed(func)
    return completed


def _discard_active_generators(
    obj: MousemapProvider,
    callback_type: Literal['move', 'drag', 'wheel'],
) -> None:
    """Close and forget all active generators of type `callback_type`.

    Code after a generator's loop does not run, so step it with an ending
    event first.
    """
    gen_dict = getattr(obj, f'_mouse_{callback_type}_gen')
    for func, gen in tuple(gen_dict.items()):
        # forget first, in case `close()` raises
        _forget_generator(obj, gen_dict, func, gen)
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            _warn_callback_failed(func)


def mouse_wheel_callbacks(
    obj: MousemapProvider, event: ReadOnlyWrapper[NapariMouseEvent]
) -> None:
    """Run mouse wheel callbacks on either layer or viewer object.

    Note that wheel callbacks can be single function callbacks, or
    generators which should have the following form:

    .. code-block:: python

        def hello_world(layer, event):
            # initial setup
            print('hello world!')
            yield

            # on subsequent scrolls
            while (some_falsifiable_condition)
                print(event.pos)
                yield

            # when done
            print('goodbye world ;(')

    Parameters
    ---------
    obj : ViewerModel or Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    completed = _step_active_generators(obj, event, 'wheel')
    _run_callbacks_and_maybe_store_generators(
        obj, event, 'wheel', ignore=completed
    )


def mouse_double_click_callbacks(
    obj: MousemapProvider, event: ReadOnlyWrapper[NapariMouseEvent]
) -> None:
    """Run mouse double_click callbacks on either layer or viewer object.

    double_click callbacks must be regular functions, not generators:

    .. code-block:: python

        def double_click_callback(layer, event):
            layer._finish_drawing()

    Parameters
    ----------
    obj : ViewerModel or Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event

    Returns
    -------
    None

    """
    # iterate through drag callback functions
    for mouse_click_func in obj.mouse_double_click_callbacks:
        # execute function to run press event code
        if inspect.isgeneratorfunction(mouse_click_func):
            raise ValueError("Double-click actions can't be generators.")
        mouse_click_func(obj, event)


def mouse_press_callbacks(
    obj: MousemapProvider, event: ReadOnlyWrapper[NapariMouseEvent]
) -> None:
    """Run mouse press callbacks on either layer or viewer object.

    Drag callbacks go through this machinery too on setup (since the first
    step of a drag is a press).

    Note that drag callbacks should have the following form:

    .. code-block:: python

        def hello_world(layer, event):
            "dragging"
            # on press
            print('hello world!')
            yield

            # on move
            while event.type == 'mouse_move':
                print(event.pos)
                yield

            # on release
            print('goodbye world ;(')

    Parameters
    ----------
    obj : ViewerModel or Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    completed = _step_active_generators(obj, event, 'drag')
    _run_callbacks_and_maybe_store_generators(
        obj, event, 'drag', ignore=completed
    )


def mouse_move_callbacks(
    obj: MousemapProvider, event: ReadOnlyWrapper[NapariMouseEvent]
) -> None:
    """Run mouse move callbacks on either layer or viewer object.

    Note that move callbacks should have the following form:

    .. code-block:: python

        def hello_world(layer, event):
            # initial setup
            print('hello world!')
            yield

            # on subsequent moves
            while (some_falsifiable_condition)
                print(event.pos)
                yield

            # when done
            print('goodbye world ;(')

    .. versionchanged:: 0.10.0
        Move callbacks are also called during drags; check
        `event.is_dragging` to skip them.

    Parameters
    ----------
    obj : ViewerModel or Layer
        Layer or Viewer object to run callbacks on
    event : NapariMouseEvent
        Mouse event
    """
    completed = _step_active_generators(obj, event, 'move')
    _run_callbacks_and_maybe_store_generators(
        obj, event, 'move', ignore=completed
    )

    if event.is_dragging:
        _step_active_generators(obj, event, 'drag')


def mouse_release_callbacks(
    obj: MousemapProvider, event: ReadOnlyWrapper[NapariMouseEvent]
) -> None:
    """Run mouse release callbacks on either layer or viewer object.

    Drag callbacks go through this machinery at the end.
    Note that drag callbacks should have the following form:


    .. code-block:: python

        def hello_world(layer, event):
            "dragging"
            # on press
            print('hello world!')
            yield

            # on move
            while event.type == 'mouse_move':
                print(event.pos)
                yield

            # on release
            print('goodbye world ;(')

    Parameters
    ----------
    obj : ViewerModel or Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    try:
        _step_active_generators(obj, event, 'drag')
    finally:
        # drags end on release, finished or not
        _discard_active_generators(obj, 'drag')


KEY_SYMBOLS = {
    'Ctrl': KeyCode.from_string('Ctrl').os_symbol(),
    'Shift': KeyCode.from_string('Shift').os_symbol(),
    'Alt': KeyCode.from_string('Alt').os_symbol(),
    'Meta': KeyCode.from_string('Meta').os_symbol(),
    'Left': KeyCode.from_string('Left').os_symbol(),
    'Right': KeyCode.from_string('Right').os_symbol(),
    'Up': KeyCode.from_string('Up').os_symbol(),
    'Down': KeyCode.from_string('Down').os_symbol(),
    'Backspace': KeyCode.from_string('Backspace').os_symbol(),
    'Delete': KeyCode.from_string('Delete').os_symbol(),
    'Tab': KeyCode.from_string('Tab').os_symbol(),
    'Escape': KeyCode.from_string('Escape').os_symbol(),
    'Return': KeyCode.from_string('Return').os_symbol(),
    'Enter': KeyCode.from_string('Enter').os_symbol(),
    'Space': KeyCode.from_string('Space').os_symbol(),
}


JOINCHAR = '+'
if sys.platform.startswith('darwin'):
    JOINCHAR = ''


class Shortcut:
    """
    Wrapper object around shortcuts,

    Mostly help to handle cross platform differences in UI:
      - whether the joiner is -,'' or something else.
      - replace the corresponding modifier with their equivalents.

    As well as integration with qt which uses a different convention with +
    instead of -.
    """

    def __init__(self, shortcut: KeyBindingLike) -> None:
        """Parameters
        ----------
        shortcut : keybinding-like
            shortcut to format
        """
        error_msg = f'`{shortcut}` does not seem to be a valid shortcut Key.'
        error = False

        try:
            self._kb = coerce_keybinding(shortcut)
        except ValueError:
            error = True
        else:
            for part in self._kb.parts:
                shortcut_key = str(part.key)
                if len(shortcut_key) > 1 and shortcut_key not in KEY_SYMBOLS:
                    error = True

        if error:
            warnings.warn(error_msg, UserWarning, stacklevel=2)

    @staticmethod
    def parse_platform(text: str) -> str:
        """
        Parse a current_platform_specific shortcut, and return a canonical
        version separated with dashes.

        This replace platform specific symbols, like ↵ by Enter,  ⌘ by Command on MacOS....
        """
        # edge case, shortcut combination where `+` is a key.
        # this should be rare as on english keyboard + is Shift-Minus.
        # but not unheard of. In those case `+` is always at the end with `++`
        # as you can't get two non-modifier keys,  or alone.
        if text == '+':
            return text
        if JOINCHAR == '+':
            text = text.replace('++', '+Plus')
            text = text.replace('+', '')
            text = text.replace('Plus', '+')
        for k, v in KEY_SYMBOLS.items():
            if text.endswith(v):
                text = text.replace(v, k)
            else:
                text = text.replace(v, k + '-')

        return text

    @property
    def qt(self) -> str:
        """Representation of the keybinding as it would appear in Qt.

        Returns
        -------
        string
            Shortcut formatted to be used with Qt.
        """
        return str(self._kb)

    @property
    def platform(self) -> str:
        """Format the given shortcut for the current platform.

        Replace Cmd, Ctrl, Meta...etc by appropriate symbols if relevant for the
        given platform.

        Returns
        -------
        string
            Shortcut formatted to be displayed on current paltform.
        """
        return self._kb.to_text(use_symbols=True, joinchar=JOINCHAR)

    def __str__(self) -> str:
        return self.platform


def get_key_bindings_summary(
    keymap: dict, col: str = 'rgb(134, 142, 147)'
) -> str:
    """Get summary of key bindings in keymap.

    Parameters
    ----------
    keymap : dict
        Dictionary of key bindings.
    col : str
        Color string in format rgb(int, int, int) used for highlighting
        keypress combination.

    Returns
    -------
    str
        String with summary of all key_bindings and their functions.
    """
    key_bindings_strs = ['<table border="0" width="100%">']
    for key in keymap:
        keycodes = [KEY_SYMBOLS.get(k, k) for k in key.split('-')]
        keycodes_str = '+'.join(
            [f"<span style='color: {col}'><b>{k}</b></span>" for k in keycodes]
        )
        key_bindings_strs.append(
            "<tr><td width='80' style='text-align: right; padding: 4px;'>"
            f"<span style='color: rgb(66, 72, 80)'>{keycodes_str}</span></td>"
            "<td style='text-align: left; padding: 4px; color: #CCC;'>"
            f'{keymap[key]}</td></tr>'
        )
    key_bindings_strs.append('</table>')
    return ''.join(key_bindings_strs)

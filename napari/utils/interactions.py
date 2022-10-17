import inspect
import sys
import warnings
from typing import List

from numpydoc.docscrape import FunctionDoc

from ..utils.key_bindings import KeyBindingLike, coerce_keybinding
from ..utils.translations import trans


def mouse_wheel_callbacks(obj, event):
    """Run mouse wheel callbacks on either layer or viewer object.

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
    ---------
    obj : ViewerModel or Layer
        Layer or Viewer object to run callbacks on
    event : Event
        Mouse event
    """
    # iterate through drag callback functions
    for mouse_wheel_func in obj.mouse_wheel_callbacks:
        # execute function to run press event code
        gen = mouse_wheel_func(obj, event)
        # if function returns a generator then try to iterate it
        if inspect.isgenerator(gen):
            try:
                next(gen)
                # now store iterated genenerator
                obj._mouse_wheel_gen[mouse_wheel_func] = gen
                # and now store event that initially triggered the press
                obj._persisted_mouse_event[gen] = event
            except StopIteration:
                pass


def mouse_double_click_callbacks(obj, event) -> None:
    """Run mouse double_click callbacks on either layer or viewer object.

    Note that unlike other press and release callback those can't be generators:

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
            raise ValueError(
                trans._(
                    "Double-click actions can't be generators.", deferred=True
                )
            )
        mouse_click_func(obj, event)


def mouse_press_callbacks(obj, event):
    """Run mouse press callbacks on either layer or viewer object.

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
    # iterate through drag callback functions
    for mouse_drag_func in obj.mouse_drag_callbacks:
        # execute function to run press event code
        gen = mouse_drag_func(obj, event)
        # if function returns a generator then try to iterate it
        if inspect.isgenerator(gen):
            try:
                next(gen)
                # now store iterated genenerator
                obj._mouse_drag_gen[mouse_drag_func] = gen
                # and now store event that initially triggered the press
                obj._persisted_mouse_event[gen] = event
            except StopIteration:
                pass


def mouse_move_callbacks(obj, event):
    """Run mouse move callbacks on either layer or viewer object.

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
    if not event.is_dragging:
        # if not dragging simply call the mouse move callbacks
        for mouse_move_func in obj.mouse_move_callbacks:
            mouse_move_func(obj, event)

    # for each drag callback get the current generator
    for func, gen in tuple(obj._mouse_drag_gen.items()):
        # save the event current event
        obj._persisted_mouse_event[gen].__wrapped__ = event
        try:
            # try to advance the generator
            next(gen)
        except StopIteration:
            # If done deleted the generator and stored event
            del obj._mouse_drag_gen[func]
            del obj._persisted_mouse_event[gen]


def mouse_release_callbacks(obj, event):
    """Run mouse release callbacks on either layer or viewer object.

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
    for func, gen in tuple(obj._mouse_drag_gen.items()):
        obj._persisted_mouse_event[gen].__wrapped__ = event
        try:
            # Run last part of the function to trigger release event
            next(gen)
        except StopIteration:
            pass
        # Finally delete the generator and stored event
        del obj._mouse_drag_gen[func]
        del obj._persisted_mouse_event[gen]


KEY_SYMBOLS = {
    'Ctrl': 'Ctrl',
    'Shift': '⇧',
    'Alt': 'Alt',
    'Meta': '⊞',
    'Left': '←',
    'Right': '→',
    'Up': '↑',
    'Down': '↓',
    'Backspace': '⌫',
    'Delete': '⌦',
    'Tab': '↹',
    'Escape': 'Esc',
    'Return': '⏎',
    'Enter': '↵',
    'Space': '␣',
}


joinchar = '+'
if sys.platform.startswith('darwin'):
    KEY_SYMBOLS.update({'Ctrl': '⌘', 'Alt': '⌥', 'Meta': '⌃'})
    joinchar = ''
elif sys.platform.startswith('linux'):
    KEY_SYMBOLS.update({'Meta': 'Super'})


def kb2mods(kb) -> List[str]:
    mods = []
    if kb.ctrl:
        mods.append('Ctrl')
    if kb.shift:
        mods.append('Shift')
    if kb.alt:
        mods.append('Alt')
    if kb.meta:
        mods.append('Meta')
    return mods


class Shortcut:
    """
    Wrapper object around shortcuts,

    Mostly help to handle cross platform differences in UI:
      - whether the joiner is -,'' or something else.
      - replace the corresponding modifier with their equivalents.

    As well as integration with qt which uses a different convention with +
    instead of -.
    """

    def __init__(self, shortcut: KeyBindingLike):
        """
        Parameters
        ----------
        shortcut : keybinding-like
            shortcut to format

        """
        error_msg = trans._(
            "{shortcut} does not seem to be a valid shortcut Key.",
            shortcut=shortcut,
        )
        error = False

        try:
            self._kb = coerce_keybinding(shortcut)
        except ValueError:
            error = True
        else:
            for part in self._kb.parts:
                shortcut_key = str(part.key)
                if (
                    len(shortcut_key) > 1
                    and shortcut_key not in KEY_SYMBOLS.keys()
                ):
                    error = True

        if error:
            warnings.warn(error_msg, UserWarning, stacklevel=2)

    @property
    def qt(self) -> str:
        return str(self._kb)

    @property
    def platform(self) -> str:
        """
        Format the given shortcut for the current platform.

        Replace Cmd, Ctrl, Meta...etc by appropriate symbols if relevant for the
        given platform.

        Returns
        -------
        string
            Shortcut formatted to be displayed on current paltform.
        """
        return ' '.join(
            joinchar.join(
                KEY_SYMBOLS.get(x, x)
                for x in (kb2mods(part) + [str(part.key)])
            )
            for part in self._kb.parts
        )

    def __str__(self):
        return self.platform


def get_key_bindings_summary(keymap, col='rgb(134, 142, 147)'):
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
        keycodes = "+".join(
            [f"<span style='color: {col}'><b>{k}</b></span>" for k in keycodes]
        )
        key_bindings_strs.append(
            "<tr><td width='80' style='text-align: right; padding: 4px;'>"
            f"<span style='color: rgb(66, 72, 80)'>{keycodes}</span></td>"
            "<td style='text-align: left; padding: 4px; color: #CCC;'>"
            f"{keymap[key]}</td></tr>"
        )
    key_bindings_strs.append('</table>')
    return ''.join(key_bindings_strs)


def get_function_summary(func):
    """Get summary of doc string of function."""
    doc = FunctionDoc(func)
    summary = ''
    for s in doc['Summary']:
        summary += s
    return summary.rstrip('.')

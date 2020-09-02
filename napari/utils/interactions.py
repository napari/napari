import inspect
import sys

import wrapt
from numpydoc.docscrape import FunctionDoc


class ReadOnlyWrapper(wrapt.ObjectProxy):
    """
    Disable item and attribute setting with the exception of  ``__wrapped__``.
    """

    def __setattr__(self, name, val):
        if name != '__wrapped__':
            raise TypeError(f'cannot set attribute {name}')
        super().__setattr__(name, val)

    def __setitem__(self, name, val):
        raise TypeError(f'cannot set item {name}')


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
        # exectute function to run press event code
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
    'Control': 'Ctrl',
    'Shift': '⇧',
    'Alt': 'Alt',
    'Option': 'Opt',
    'Meta': '⊞',
    'Left': '←',
    'Right': '→',
    'Up': '↑',
    'Down': '↓',
    'Backspace': '⌫',
    'Tab': '↹',
    'Escape': 'Esc',
    'Return': '⏎',
    'Enter': '↵',
}


if sys.platform.startswith('darwin'):
    KEY_SYMBOLS.update(
        {'Control': '⌘', 'Alt': '⌥', 'Option': '⌥', 'Meta': '⌃'}
    )
elif sys.platform.startswith('linux'):
    KEY_SYMBOLS.update({'Meta': 'Super'})


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
    key_bindings_str : str
        String with summary of all key_bindings and their functions.
    """
    key_bindings_str = '<table border="0" width="100%">'
    for key in keymap:
        keycodes = [KEY_SYMBOLS.get(k, k) for k in key.split('-')]
        keycodes = "+".join(
            [f"<span style='color: {col}'><b>{k}</b></span>" for k in keycodes]
        )
        key_bindings_str += (
            "<tr><td width='80' style='text-align: right; padding: 4px;'>"
            f"<span style='color: rgb(66, 72, 80)'>{keycodes}</span></td>"
            "<td style='text-align: left; padding: 4px; color: #CCC;'>"
            f"{get_function_summary(keymap[key])}</td></tr>"
        )
    key_bindings_str += '</table>'
    return key_bindings_str


def get_function_summary(func):
    """Get summary of doc string of function."""
    doc = FunctionDoc(func)
    summary = ''
    for s in doc['Summary']:
        summary += s
    return summary.rstrip('.')

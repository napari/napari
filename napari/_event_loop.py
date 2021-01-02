from contextlib import contextmanager

try:
    from ._qt.qt_event_loop import gui_qt

# qtpy raises a RuntimeError if no Qt bindings can be found
except (ImportError, RuntimeError) as e:

    exc = e

    @contextmanager
    def gui_qt(**kwargs):
        raise exc

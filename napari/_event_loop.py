from contextlib import contextmanager

try:
    from ._qt.qt_event_loop import gui_qt

except ImportError as e:

    exc = e

    @contextmanager
    def gui_qt(**kwargs):
        raise exc

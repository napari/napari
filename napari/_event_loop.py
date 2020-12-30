from contextlib import contextmanager

try:
    from ._qt.qt_event_loop import gui_qt as event_loop

except ImportError as e:

    exc = e

    @contextmanager
    def event_loop(**kwargs):
        raise exc

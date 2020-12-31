try:
    from ._qt.qt_event_loop import gui_qt, run_app

except ImportError as e:

    exc = e

    def gui_qt(**kwargs):
        raise exc

    def run_app(**kwargs):
        raise exc

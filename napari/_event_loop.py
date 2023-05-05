try:
    from napari._qt.qt_event_loop import gui_qt, run

# qtpy raises a RuntimeError if no Qt bindings can be found
except (ImportError, RuntimeError) as e:
    exc = e

    def gui_qt(**kwargs):
        raise exc

    def run(**kwargs):
        raise exc

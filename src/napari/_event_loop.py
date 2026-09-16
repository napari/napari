try:
    from napari._qt.qt_event_loop import run

# qtpy raises a RuntimeError if no Qt bindings can be found
except (ImportError, RuntimeError) as e:
    exc = e

    def run(
        *,
        force: bool = False,
        gui_exceptions: bool = False,
        max_loop_level: int = 1,
        _func_name: str = 'run',
    ) -> None:
        raise exc

from functools import lru_cache
from typing import Optional

from napari._qt.qt_viewer import QtViewer

# Submodules should be able to import from most modules, so to
# avoid circular imports, don't import submodules at the top level here,
# import them inside the init_qactions function.


@lru_cache  # only call once
def init_qactions():
    from itertools import chain

    from napari._app_model import get_app

    from ...qt_main_window import Window, _QtMainWindow
    from ...qt_viewer import QtViewer
    from ._file import FILE_ACTIONS

    # Qt-specific providers/processors
    def _provide_window() -> Optional[Window]:
        if _qmainwin := _QtMainWindow.current():
            return _qmainwin._window

    def _provide_qtviewer() -> Optional[QtViewer]:
        if _qmainwin := _QtMainWindow.current():
            return _qmainwin._qt_viewer

    # update the namespace with the Qt-specific types/providers/processors
    app = get_app()
    store = app.injection_store
    store.namespace = {
        **store.namespace,
        'Window': Window,
        'QtViewer': QtViewer,
    }
    store.register(providers=[(_provide_window,), (_provide_qtviewer,)])

    # register actions
    for action in chain(FILE_ACTIONS):
        app.register_action(action)

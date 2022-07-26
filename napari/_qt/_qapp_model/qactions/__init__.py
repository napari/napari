from functools import lru_cache
from typing import Optional


@lru_cache  # only call once
def init_qactions():
    from itertools import chain

    from napari._app_model import get_app

    from ...qt_main_window import Window, _QtMainWindow
    from ._view import VIEW_ACTIONS

    # Qt-specific providers/processors
    def _provide_window() -> Optional[Window]:
        if _qmainwin := _QtMainWindow.current():
            return _qmainwin._window

    # update the namespace with the Qt-specific types/providers/processors
    app = get_app()
    store = app.injection_store
    store.namespace = {**store.namespace, 'Window': Window}
    store.register_provider(_provide_window)

    # register actions
    for action in chain(VIEW_ACTIONS):
        app.register_action(action)

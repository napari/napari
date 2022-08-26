from functools import lru_cache
from typing import Optional

# Submodules should be able to import from most modules, so to
# avoid circular imports, don't import submodules at the top level here,
# import them inside the init_qactions function.


@lru_cache  # only call once
def init_qactions():
    from itertools import chain

    from napari._app_model import get_app

    from ...qt_main_window import Window, _QtMainWindow
    from ._plugins import PLUGINS_ACTIONS

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
    for action in chain(PLUGINS_ACTIONS):
        app.register_action(action)

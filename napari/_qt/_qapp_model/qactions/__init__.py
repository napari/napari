from itertools import chain
from functools import lru_cache
from typing import Optional

# Submodules should be able to import from most modules, so to
# avoid circular imports, don't import submodules at the top level here,
# import them inside the init_qactions function.


@lru_cache  # only call once
def init_qactions() -> None:
    """Initialize all Qt-based Actions with app-model
    This function will be called in _QtMainWindow.__init__().  It should only
    be called once (hence the lru_cache decorator).
    It is responsible for:
    - injecting Qt-specific names into the application injection_store namespace
      (this is what allows functions to be declared with annotations like
      `def foo(window: Window)` or `def foo(qt_viewer: QtViewer)`)
    - registering provider functions for the names added to the namespace
    - registering Qt-dependent actions with app-model (i.e. Q_* actions).
    """

    from ...._app_model import get_app
    from ...qt_main_window import Window, _QtMainWindow
    from ...qt_viewer import QtViewer
    from ._plugins import Q_PLUGINS_ACTIONS

    # update the namespace with the Qt-specific types/providers/processors
    app = get_app()
    store = app.injection_store
    store.namespace = {
        **store.namespace,
        'Window': Window,
        'QtViewer': QtViewer,
    }

    # Qt-specific providers/processors
    @store.register_provider
    def _provide_window() -> Optional[Window]:
        if _qmainwin := _QtMainWindow.current():
            return _qmainwin._window

    @store.register_provider
    def _provide_qt_viewer() -> Optional[QtViewer]:
        if _qmainwin := _QtMainWindow.current():
            return _qmainwin._qt_viewer

    # register actions
    for action in chain(Q_PLUGINS_ACTIONS):
        app.register_action(action)

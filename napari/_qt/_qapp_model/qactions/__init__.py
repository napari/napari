from __future__ import annotations

from functools import lru_cache
from itertools import chain

from napari._qt._qapp_model.injection._qprocessors import QPROCESSORS
from napari._qt._qapp_model.injection._qproviders import QPROVIDERS

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
    - registering Qt-dependent actions with app-model (i.e. Q_*_ACTIONS actions).
    """
    from napari._app_model import get_app
    from napari._qt._qapp_model.qactions._debug import (
        DEBUG_SUBMENUS,
        Q_DEBUG_ACTIONS,
    )
    from napari._qt._qapp_model.qactions._file import (
        FILE_SUBMENUS,
        Q_FILE_ACTIONS,
    )
    from napari._qt._qapp_model.qactions._help import Q_HELP_ACTIONS
    from napari._qt._qapp_model.qactions._layerlist_context import (
        Q_LAYERLIST_CONTEXT_ACTIONS,
    )
    from napari._qt._qapp_model.qactions._layers_actions import (
        LAYERS_ACTIONS,
        LAYERS_SUBMENUS,
    )
    from napari._qt._qapp_model.qactions._plugins import Q_PLUGINS_ACTIONS
    from napari._qt._qapp_model.qactions._view import (
        Q_VIEW_ACTIONS,
        VIEW_SUBMENUS,
    )
    from napari._qt._qapp_model.qactions._window import Q_WINDOW_ACTIONS
    from napari._qt.qt_main_window import Window
    from napari._qt.qt_viewer import QtViewer

    # update the namespace with the Qt-specific types/providers/processors
    app = get_app()
    store = app.injection_store
    store.namespace = {
        **store.namespace,
        'Window': Window,
        'QtViewer': QtViewer,
    }

    # Qt-specific providers/processors
    app.injection_store.register(
        processors=QPROCESSORS,
        providers=QPROVIDERS,
    )

    # register menubar actions
    app.register_actions(
        chain(
            Q_DEBUG_ACTIONS,
            Q_FILE_ACTIONS,
            Q_HELP_ACTIONS,
            Q_PLUGINS_ACTIONS,
            Q_VIEW_ACTIONS,
            LAYERS_ACTIONS,
            Q_LAYERLIST_CONTEXT_ACTIONS,
            Q_WINDOW_ACTIONS,
        )
    )

    # register menubar submenus
    app.menus.append_menu_items(
        chain(FILE_SUBMENUS, VIEW_SUBMENUS, DEBUG_SUBMENUS, LAYERS_SUBMENUS)
    )

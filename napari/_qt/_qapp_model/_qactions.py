from functools import lru_cache
from typing import List, Optional

from app_model.types import Action, KeyCode, KeyMod, StandardKeyBinding

from ..._app.constants import CommandId, MenuGroup, MenuId
from ...settings import get_settings
from ...utils.translations import trans


def _tooltip_visibility_toggle():
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


@lru_cache  # only call once
def _init_qactions():
    from napari._app import app

    from ..qt_main_window import Window, _QtMainWindow
    from ..qt_viewer import QtViewer

    VIEW_ACTIONS: List[Action] = [
        Action(
            id=CommandId.TOGGLE_FULLSCREEN,
            title=CommandId.TOGGLE_FULLSCREEN.title,
            menus=[
                {
                    'id': MenuId.MENUBAR_VIEW,
                    'group': MenuGroup.NAVIGATION,
                    'order': 1,
                }
            ],
            callback=Window._toggle_fullscreen,
            keybindings=[StandardKeyBinding.FullScreen],
        ),
        Action(
            id=CommandId.TOGGLE_MENUBAR,
            title=CommandId.TOGGLE_MENUBAR.title,
            menus=[
                {
                    'id': MenuId.MENUBAR_VIEW,
                    'group': MenuGroup.NAVIGATION,
                    'order': 2,
                    'when': 'not is_mac',
                }
            ],
            callback=Window._toggle_menubar_visible,
            keybindings=[
                {
                    'win': KeyMod.CtrlCmd | KeyCode.KeyM,
                    'linux': KeyMod.CtrlCmd | KeyCode.KeyM,
                }
            ],
            enablement='not is_mac',
            status_tip=trans._('Show/Hide Menubar'),
        ),
        Action(
            id=CommandId.TOGGLE_PLAY,
            title=CommandId.TOGGLE_PLAY.title,
            menus=[
                {
                    'id': MenuId.MENUBAR_VIEW,
                    'group': MenuGroup.NAVIGATION,
                    'order': 3,
                }
            ],
            callback=Window._toggle_play,
            keybindings=[
                {'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyP}
            ],
        ),
        Action(
            id=CommandId.TOGGLE_OCTREE_CHUNK_OUTLINES,
            title=CommandId.TOGGLE_OCTREE_CHUNK_OUTLINES.title,
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 1}
            ],
            callback=QtViewer._toggle_chunk_outlines,
            # this used to have a keybinding of Ctrl+Alt+O, but that conflicts with
            # Open files as stack
            enablement='settings_experimental_octree',  # TODO
        ),
        # TODO: this could be made into a toggle setting Action subclass
        Action(
            id=CommandId.TOGGLE_LAYER_TOOLTIPS,
            title=CommandId.TOGGLE_LAYER_TOOLTIPS.title,
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 10}
            ],
            callback=_tooltip_visibility_toggle,
            toggled='settings_appearance_layer_tooltip_visibility',  # TODO
        ),
        Action(
            id=CommandId.TOGGLE_ACTIVITY_DOCK,
            title=CommandId.TOGGLE_ACTIVITY_DOCK.title,
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 11}
            ],
            # callback=Window._status_bar._toggle_activity_dock,
            callback='',
            toggled='settings_appearance_activity_dock_visible',  # TODO
            # 'checked': window._qt_window._activity_dialog.isVisible(),
        ),
    ]

    for action in VIEW_ACTIONS:
        app.register_action(action)

    def _provide_window() -> Optional[Window]:
        if _qmainwin := _QtMainWindow.current():
            return _qmainwin._window

    ns = app.injection_store.namespace
    ns.update({'Window': Window})
    app.injection_store.namespace = ns
    app.injection_store.register_provider(_provide_window)

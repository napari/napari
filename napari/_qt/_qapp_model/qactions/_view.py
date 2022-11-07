"""Actions related to the 'View' menu that require Qt.

'View' actions that do not require Qt should go in
`napari/_app_model/actions/_view_actions.py`.
"""

import sys
from typing import List

from app_model.types import (
    Action,
    KeyCode,
    KeyMod,
    StandardKeyBinding,
    ToggleRule,
)

from ...._app_model.constants import CommandId, MenuGroup, MenuId
from ....settings import get_settings
from ....utils.translations import trans
from ...qt_main_window import Window
from ...qt_viewer import QtViewer


def _toggle_activity_dock(window: Window):
    window._status_bar._toggle_activity_dock()


def _get_current_activity_dock(window: Window):
    return window._qt_window._activity_dialog.isVisible()


Q_VIEW_ACTIONS: List[Action] = [
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
                'when': sys.platform != 'darwin',
            }
        ],
        callback=Window._toggle_menubar_visible,
        keybindings=[
            {
                'win': KeyMod.CtrlCmd | KeyCode.KeyM,
                'linux': KeyMod.CtrlCmd | KeyCode.KeyM,
            }
        ],
        # TODO: add is_mac global context keys (rather than boolean here)
        enablement=sys.platform != 'darwin',
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
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyP}],
    ),
    Action(
        id=CommandId.TOGGLE_OCTREE_CHUNK_OUTLINES,
        title=CommandId.TOGGLE_OCTREE_CHUNK_OUTLINES.title,
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.RENDER,
                'order': 1,
                'when': get_settings().experimental.octree,
            }
        ],
        callback=QtViewer._toggle_chunk_outlines,
        enablement=get_settings().experimental.octree,
        # this used to have a keybinding of Ctrl+Alt+O, but that conflicts with
        # Open files as stack
    ),
    Action(
        id=CommandId.TOGGLE_ACTIVITY_DOCK,
        title=CommandId.TOGGLE_ACTIVITY_DOCK.title,
        menus=[
            {'id': MenuId.MENUBAR_VIEW, 'group': MenuGroup.RENDER, 'order': 11}
        ],
        callback=_toggle_activity_dock,
        toggled=ToggleRule(get_current=_get_current_activity_dock),
    ),
]

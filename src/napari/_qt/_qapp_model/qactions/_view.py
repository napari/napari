"""Qt 'View' menu Actions."""

import sys

from app_model.types import (
    Action,
    KeyCode,
    KeyMod,
    StandardKeyBinding,
    ToggleRule,
)

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari.utils.translations import trans


# View actions
def _toggle_activity_dock(window: Window):
    window._status_bar._toggle_activity_dock()


def _get_current_fullscreen_status(window: Window):
    return window._qt_window.isFullScreen()


def _get_current_menubar_status(window: Window):
    return window._qt_window._toggle_menubar_visibility


def _get_current_play_status(qt_viewer: QtViewer):
    return bool(qt_viewer.dims.is_playing)


def _get_current_activity_dock_status(window: Window):
    return window._qt_window._activity_dialog.isVisible()


Q_VIEW_ACTIONS: list[Action] = [
    Action(
        id='napari.window.view.toggle_fullscreen',
        title=trans._('Toggle Full Screen'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 1,
            }
        ],
        callback=Window._toggle_fullscreen,
        keybindings=[StandardKeyBinding.FullScreen],
        toggled=ToggleRule(get_current=_get_current_fullscreen_status),
    ),
    Action(
        id='napari.window.view.toggle_command_palette',
        title=trans._('Command Palette'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 2,
            }
        ],
        callback=Window._toggle_command_palette,
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyP}
        ],
    ),
    Action(
        id='napari.window.view.toggle_menubar',
        title=trans._('Toggle Menubar Visibility'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 3,
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
        toggled=ToggleRule(get_current=_get_current_menubar_status),
    ),
    Action(
        id='napari.window.view.toggle_play',
        title=trans._('Toggle Play'),
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.NAVIGATION,
                'order': 3,
            }
        ],
        callback=Window._toggle_play,
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyP}],
        toggled=ToggleRule(get_current=_get_current_play_status),
    ),
    Action(
        id='napari.window.view.toggle_activity_dock',
        title=trans._('Toggle Activity Dock'),
        menus=[
            {'id': MenuId.MENUBAR_VIEW, 'group': MenuGroup.RENDER, 'order': 11}
        ],
        callback=_toggle_activity_dock,
        toggled=ToggleRule(get_current=_get_current_activity_dock_status),
    ),
]

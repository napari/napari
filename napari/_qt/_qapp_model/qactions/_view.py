"""Actions related to the 'View' menu that require Qt.

'View' actions that do not require Qt should be placed under
`napari/_app_model/actions/` in a `_view_actions.py` file.
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

from napari._app_model.actions._toggle_action import ViewerToggleAction
from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari.settings import get_settings
from napari.utils.translations import trans


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


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


Q_VIEW_ACTIONS: List[Action] = [
    Action(
        id=CommandId.TOGGLE_FULLSCREEN,
        title=CommandId.TOGGLE_FULLSCREEN.command_title,
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
        id=CommandId.TOGGLE_MENUBAR,
        title=CommandId.TOGGLE_MENUBAR.command_title,
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
        toggled=ToggleRule(get_current=_get_current_menubar_status),
    ),
    Action(
        id=CommandId.TOGGLE_PLAY,
        title=CommandId.TOGGLE_PLAY.command_title,
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
        id=CommandId.TOGGLE_ACTIVITY_DOCK,
        title=CommandId.TOGGLE_ACTIVITY_DOCK.command_title,
        menus=[
            {'id': MenuId.MENUBAR_VIEW, 'group': MenuGroup.RENDER, 'order': 11}
        ],
        callback=_toggle_activity_dock,
        toggled=ToggleRule(get_current=_get_current_activity_dock_status),
    ),
    # TODO: this could be made into a toggle setting Action subclass
    # using a similar pattern to the above ViewerToggleAction classes
    Action(
        id=CommandId.TOGGLE_LAYER_TOOLTIPS,
        title=CommandId.TOGGLE_LAYER_TOOLTIPS.command_title,
        menus=[
            {
                'id': MenuId.MENUBAR_VIEW,
                'group': MenuGroup.RENDER,
                'order': 10,
            }
        ],
        callback=_tooltip_visibility_toggle,
        toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
    ),
]

MENUID_DICT = {'axes': MenuId.VIEW_AXES, 'scale_bar': MenuId.VIEW_SCALEBAR}

for cmd, viewer_attr, sub_attr in (
    (CommandId.TOGGLE_VIEWER_AXES, 'axes', 'visible'),
    (CommandId.TOGGLE_VIEWER_AXES_COLORED, 'axes', 'colored'),
    (CommandId.TOGGLE_VIEWER_AXES_LABELS, 'axes', 'labels'),
    (CommandId.TOGGLE_VIEWER_AXES_DASHED, 'axes', 'dashed'),
    (CommandId.TOGGLE_VIEWER_AXES_ARROWS, 'axes', 'arrows'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR, 'scale_bar', 'visible'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR_COLORED, 'scale_bar', 'colored'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR_TICKS, 'scale_bar', 'ticks'),
):
    Q_VIEW_ACTIONS.append(
        ViewerToggleAction(
            id=cmd,
            title=cmd.command_title,
            viewer_attribute=viewer_attr,
            sub_attribute=sub_attr,
            menus=[{'id': MENUID_DICT[viewer_attr]}],
        )
    )

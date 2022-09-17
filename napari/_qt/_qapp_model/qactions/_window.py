"""Defines window menu actions."""

from typing import List

from app_model.types import Action

from ...._app_model.constants import CommandId, MenuGroup, MenuId
from ....utils.translations import trans
from ...qt_main_window import Window


def _toggle_console(window: Window):
    window._qt_viewer.dockConsole.setVisible(
        not window._qt_viewer.dockConsole.isVisible()
    )


def _toggle_layer_controls(window: Window):
    window._qt_viewer.dockLayerControls.setVisible(
        not window._qt_viewer.dockLayerControls.isVisible()
    )


def _toggle_layer_list(window: Window):
    window._qt_viewer.dockLayerList.setVisible(
        not window._qt_viewer.dockLayerList.isVisible()
    )


WINDOW_ACTIONS: List[Action] = [
    Action(
        id=CommandId.TOGGLE_CONSOLE,
        title=CommandId.TOGGLE_CONSOLE.title,
        menus=[
            {
                'id': MenuId.MENUBAR_WINDOW,
                'group': MenuGroup.NAVIGATION,
                'order': 1,
            }
        ],
        callback=_toggle_console,
        status_tip=trans._('Toggle console panel'),
    ),
    Action(
        id=CommandId.TOGGLE_LAYER_CONTROLS,
        title=CommandId.TOGGLE_LAYER_CONTROLS.title,
        menus=[
            {
                'id': MenuId.MENUBAR_WINDOW,
                'group': MenuGroup.NAVIGATION,
                'order': 2,
            }
        ],
        callback=_toggle_layer_controls,
        status_tip=trans._('Toggle layer controls panel'),
    ),
    Action(
        id=CommandId.TOGGLE_LAYER_LIST,
        title=CommandId.TOGGLE_LAYER_LIST.title,
        menus=[
            {
                'id': MenuId.MENUBAR_WINDOW,
                'group': MenuGroup.NAVIGATION,
                'order': 3,
            }
        ],
        callback=_toggle_layer_list,
        status_tip=trans._('Toggle layer list panel'),
    ),
]

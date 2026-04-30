"""Qt 'Window' menu Actions."""

from app_model.types import Action

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt._qapp_model.qactions._toggle_action import (
    DockWidgetToggleAction,
)


Q_WINDOW_ACTIONS: list[Action] = []

toggle_action_details = [
    (
        'napari.window.toggle_window_console',
        'Console',
        'dockConsole',
        'Toggle console panel',
    ),
    (
        'napari.window.toggle_layer_controls',
        'Layer Controls',
        'dockLayerControls',
        'Toggle layer controls panel',
    ),
    (
        'napari.window.toggle_layer_list',
        'Layer List',
        'dockLayerList',
        'Toggle layer list panel',
    ),
]
for cmd_id, cmd_title, dock_widget, status_tip in toggle_action_details:
    Q_WINDOW_ACTIONS.append(
        DockWidgetToggleAction(
            id=cmd_id,
            title=cmd_title,
            dock_widget=dock_widget,
            menus=[
                {
                    'id': MenuId.MENUBAR_WINDOW,
                    'group': MenuGroup.NAVIGATION,
                }
            ],
            status_tip=status_tip,
        )
    )

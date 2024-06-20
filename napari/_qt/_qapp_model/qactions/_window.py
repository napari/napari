"""Qt 'Window' menu Actions."""

from app_model.types import Action

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt._qapp_model.qactions._toggle_action import (
    DockWidgetToggleAction,
)
from napari.utils.translations import trans

Q_WINDOW_ACTIONS: list[Action] = []

toggle_action_details = [
    (
        'napari:window:window:toggle_window_console',
        trans._('Console'),
        'dockConsole',
        trans._('Toggle console panel'),
    ),
    (
        'napari:window:window:toggle_layer_controls',
        trans._('Layer Controls'),
        'dockLayerControls',
        trans._('Toggle layer controls panel'),
    ),
    (
        'napari:window:window:toggle_layer_list',
        trans._('Layer List'),
        'dockLayerList',
        trans._('Toggle layer list panel'),
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

"""Defines plugins menu actions."""

from typing import List

from app_model.types import Action

from ...._app_model.constants import CommandId, MenuGroup, MenuId
from ....utils.translations import trans
from ...qt_main_window import Window

Q_PLUGINS_ACTIONS: List[Action] = [
    Action(
        id=CommandId.DLG_PLUGIN_INSTALL,
        title=CommandId.DLG_PLUGIN_INSTALL.title,
        menus=[
            {
                'id': MenuId.MENUBAR_PLUGINS,
                'group': MenuGroup.NAVIGATION,
                'order': 1,
            }
        ],
        callback=,
    ),
    Action(
        id=CommandId.DLG_PLUGIN_ERR,
        title=CommandId.DLG_PLUGIN_ERR.title,
        menus=[
            {
                'id': MenuId.MENUBAR_PLUGINS,
                'group': MenuGroup.NAVIGATION,
                'order': 2,
            }
        ],
        callback=_show_layer_controls,
    ),
]

from typing import List

from app_model.types import Action

from ...._app_model.constants import CommandId, MenuGroup, MenuId

WINDOW_ACTIONS: List[Action] = [
    Action(
        id=CommandId.TOGGLE_WINDOW_CONSOLE,
        title=CommandId.TOGGLE_WINDOW_CONSOLE.title,
        menus=[
            {
                'id': MenuId.MENUBAR_WINDOW,
                'group': MenuGroup.NAVIGATION,
                'order': 1,
            }
        ],
        callback=,
        keybindings=,
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
        callback=,
        keybindings=
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
        callback=,
        keybindings=
    ),
]
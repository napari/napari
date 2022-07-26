from typing import List

from app_model.types import Action, KeyCode, KeyMod

from ...._app_model.constants import CommandId, MenuGroup, MenuId
from ....utils.translations import trans
from ...qt_main_window import Window

FILE_ACTIONS: List[Action] = [
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
]

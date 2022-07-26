from typing import List

from app_model.types import Action, KeyCode, KeyMod, StandardKeyBinding

from ...._app_model.constants import CommandId, MenuGroup, MenuId

# from ....utils.translations import trans
# from ...qt_main_window import Window
from ...qt_viewer import QtViewer

FILE_ACTIONS: List[Action] = [
    Action(
        id=CommandId.DLG_OPEN_FILES,
        title=CommandId.DLG_OPEN_FILES.title,
        callback=QtViewer._open_files_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[StandardKeyBinding.Open],
    ),
    Action(
        id=CommandId.DLG_OPEN_FILES_AS_STACK,
        title=CommandId.DLG_OPEN_FILES_AS_STACK.title,
        callback=QtViewer._open_files_dialog_as_stack_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyO}],
    ),
    Action(
        id=CommandId.DLG_OPEN_FOLDER,
        title=CommandId.DLG_OPEN_FOLDER.title,
        callback=QtViewer._open_files_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyO}
        ],
    ),
]

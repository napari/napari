from typing import List

from app_model.types import Action, KeyCode, KeyMod, StandardKeyBinding

from ...._app_model.constants import CommandId, MenuGroup, MenuId
from ....utils.translations import trans
from ...qt_main_window import Window
from ...qt_viewer import QtViewer


def _restart(window: Window):
    window._qt_window.restart()


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
    Action(
        id=CommandId.DLG_SHOW_PREFERENCES,
        title=CommandId.DLG_SHOW_PREFERENCES.title,
        callback=Window._open_preferences_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '2_'}],
        keybindings=[StandardKeyBinding.Preferences],
    ),
    # TODO!
    # this action would conflict because having the same id as below... but
    # we could check whether the args are the same and if not, then allow registration
    # Action(
    #     id=CommandId.DLG_SAVE_LAYERS,
    #     title=CommandId.DLG_SAVE_LAYERS.title,
    #     callback=QtViewer._save_layers_dialog,
    #     kwargs={'selected': True},  # <<<<< TODO!
    #     menus=[{'id': MenuId.MENUBAR_FILE, 'group': '3_'}],
    #     keybindings=[StandardKeyBinding.Save],
    #     enablement='num_selected_layers > 0'
    # ),
    Action(
        id=CommandId.DLG_SAVE_LAYERS,
        title=trans._('Save All Layers...'),
        callback=QtViewer._save_layers_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '3_'}],
        # Conflict: ctrl+shift+s typically means "save as ..."
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyS}
        ],
        enablement='num_selected_layers > 0',
    ),
    Action(
        id=CommandId.DLG_SAVE_CANVAS_SCREENSHOT,
        title=CommandId.DLG_SAVE_CANVAS_SCREENSHOT.title,
        callback=QtViewer._screenshot_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '3_'}],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyS}],
        status_tip=trans._('Save screenshot of current display, default .png'),
    ),
    Action(
        id=CommandId.DLG_SAVE_VIEWER_SCREENSHOT,
        title=CommandId.DLG_SAVE_VIEWER_SCREENSHOT.title,
        callback=Window._screenshot_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '3_'}],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyS}],
        status_tip=trans._('Save screenshot of current display, default .png'),
    ),
    Action(
        id=CommandId.COPY_CANVAS_SCREENSHOT,
        title=CommandId.COPY_CANVAS_SCREENSHOT.title,
        callback=QtViewer.clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '3_'}],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyC}],
        status_tip=trans._(
            'Copy screenshot of current display to the clipboard'
        ),
    ),
    Action(
        id=CommandId.COPY_VIEWER_SCREENSHOT,
        title=CommandId.COPY_VIEWER_SCREENSHOT.title,
        callback=Window.clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '3_'}],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyC}],
        status_tip=trans._(
            'Copy screenshot of current display with the viewer to the clipboard'
        ),
    ),
    Action(
        id=CommandId.DLG_CLOSE,
        title=CommandId.DLG_CLOSE.title,
        callback=Window._close_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '4_'}],
        keybindings=[StandardKeyBinding.Close],
    ),
    Action(
        id=CommandId.RESTART,
        title=CommandId.RESTART.title,
        callback=_restart,
        menus=[
            {'id': MenuId.MENUBAR_FILE, 'group': '4_', 'when': 'is_bundle'}
        ],
    ),
    Action(
        id=CommandId.DLG_QUIT,
        title=CommandId.DLG_QUIT.title,
        callback=Window._quit_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': '4_'}],
    ),
]

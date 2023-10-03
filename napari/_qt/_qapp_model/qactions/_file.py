import sys
from pathlib import Path
from typing import List

from app_model.types import Action, KeyCode, KeyMod, StandardKeyBinding

from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._app_model.context import (
    LayerListContextKeys as LLCK,
    LayerListSelectionContextKeys as LLSCK,
)
from napari._qt.qt_main_window import Window
from napari._qt.qt_viewer import QtViewer
from napari.utils.translations import trans


def _open_files_with_plugin(qt_viewer: QtViewer):
    qt_viewer._open_files_dialog(choose_plugin=True)


def _open_files_as_stack_with_plugin(qt_viewer: QtViewer):
    qt_viewer._open_files_dialog_as_stack_dialog(choose_plugin=True)


def _open_folder_with_plugin(qt_viewer: QtViewer):
    qt_viewer._open_folder_dialog(choose_plugin=True)


def _save_selected_layers(qt_viewer: QtViewer):
    qt_viewer._save_layers_dialog(selected=True)


def _restart(window: Window):
    window._qt_window.restart()


def _close_window(window: Window):
    window._qt_window.close(quit_app=False, confirm_need=True)


def _close_app(window: Window):
    window._qt_window.close(quit_app=True, confirm_need=True)


Q_FILE_ACTIONS: List[Action] = [
    Action(
        id=CommandId.DLG_OPEN_FILES,
        title=CommandId.DLG_OPEN_FILES.command_title,
        callback=QtViewer._open_files_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[StandardKeyBinding.Open],
    ),
    Action(
        id=CommandId.DLG_OPEN_FILES_AS_STACK,
        title=CommandId.DLG_OPEN_FILES_AS_STACK.command_title,
        callback=QtViewer._open_files_dialog_as_stack_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyO}],
    ),
    Action(
        id=CommandId.DLG_OPEN_FOLDER,
        title=CommandId.DLG_OPEN_FOLDER.command_title,
        callback=QtViewer._open_files_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyO}
        ],
    ),
    Action(
        id=CommandId.DLG_OPEN_FILES_WITH_PLUGIN,
        title=CommandId.DLG_OPEN_FILES_WITH_PLUGIN.command_title,
        callback=_open_files_with_plugin,
        menus=[
            {'id': MenuId.FILE_OPEN_WITH_PLUGIN, 'group': MenuGroup.NAVIGATION}
        ],
    ),
    Action(
        id=CommandId.DLG_OPEN_FILES_AS_STACK_WITH_PLUGIN,
        title=CommandId.DLG_OPEN_FILES_AS_STACK_WITH_PLUGIN.command_title,
        callback=_open_files_as_stack_with_plugin,
        menus=[
            {'id': MenuId.FILE_OPEN_WITH_PLUGIN, 'group': MenuGroup.NAVIGATION}
        ],
    ),
    Action(
        id=CommandId.DLG_OPEN_FOLDER_WITH_PLUGIN,
        title=CommandId.DLG_OPEN_FOLDER_WITH_PLUGIN.command_title,
        callback=_open_folder_with_plugin,
        menus=[
            {'id': MenuId.FILE_OPEN_WITH_PLUGIN, 'group': MenuGroup.NAVIGATION}
        ],
    ),
    Action(
        id=CommandId.DLG_SHOW_PREFERENCES,
        title=CommandId.DLG_SHOW_PREFERENCES.command_title,
        callback=Window._open_preferences_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.PREFERENCES}],
        # TODO: revert to `StandardKeyBinding.Preferences` after app-model>0.2.0
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.Comma}],
    ),
    # TODO:
    # If app-model supports a `kwargs` field (see:
    # https://github.com/pyapp-kit/app-model/issues/52)
    # it may allow registration of the same `id` when args are different and
    # we can re-use `DLG_SAVE_LAYERS` below.
    Action(
        id=CommandId.DLG_SAVE_SELECTED_LAYERS,
        title=CommandId.DLG_SAVE_SELECTED_LAYERS.command_title,
        callback=_save_selected_layers,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[StandardKeyBinding.Save],
        enablement=(LLSCK.num_selected_layers > 0),
    ),
    Action(
        id=CommandId.DLG_SAVE_LAYERS,
        title=CommandId.DLG_SAVE_LAYERS.command_title,
        callback=QtViewer._save_layers_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyS}],
        enablement=(LLCK.num_layers > 0),
    ),
    Action(
        id=CommandId.DLG_SAVE_CANVAS_SCREENSHOT,
        title=CommandId.DLG_SAVE_CANVAS_SCREENSHOT.command_title,
        callback=QtViewer._screenshot_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyS}],
        status_tip=trans._('Save screenshot of current display, default .png'),
    ),
    Action(
        id=CommandId.DLG_SAVE_VIEWER_SCREENSHOT,
        title=CommandId.DLG_SAVE_VIEWER_SCREENSHOT.command_title,
        callback=Window._screenshot_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyS}],
        status_tip=trans._('Save screenshot of current display, default .png'),
    ),
    Action(
        id=CommandId.COPY_CANVAS_SCREENSHOT,
        title=CommandId.COPY_CANVAS_SCREENSHOT.command_title,
        callback=QtViewer.clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyC}],
        status_tip=trans._(
            'Copy screenshot of current display to the clipboard'
        ),
    ),
    Action(
        id=CommandId.COPY_VIEWER_SCREENSHOT,
        title=CommandId.COPY_VIEWER_SCREENSHOT.command_title,
        callback=Window.clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyC}],
        status_tip=trans._(
            'Copy screenshot of current display with the viewer to the clipboard'
        ),
    ),
    Action(
        id=CommandId.DLG_CLOSE,
        title=CommandId.DLG_CLOSE.command_title,
        callback=_close_window,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.CLOSE}],
        keybindings=[StandardKeyBinding.Close],
    ),
    Action(
        id=CommandId.RESTART,
        title=CommandId.RESTART.command_title,
        callback=_restart,
        menus=[
            {
                'id': MenuId.MENUBAR_FILE,
                'group': MenuGroup.CLOSE,
                'when': (
                    Path(sys.executable).parent / ".napari_is_bundled"
                ).exists(),
            }
        ],
    ),
    Action(
        id=CommandId.DLG_QUIT,
        title=CommandId.DLG_QUIT.command_title,
        callback=_close_app,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.CLOSE}],
    ),
]

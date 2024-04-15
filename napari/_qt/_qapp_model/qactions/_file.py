import sys
from pathlib import Path

from app_model.types import Action, KeyCode, KeyMod, StandardKeyBinding

from napari._app_model.constants import MenuGroup, MenuId
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


Q_FILE_ACTIONS: list[Action] = [
    Action(
        id='napari.window.file._image_from_clipboard',
        title=trans._('New Image from Clipboard'),
        callback=QtViewer._image_from_clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyCode.KeyN}],
    ),
    Action(
        id='napari.window.file.open_files_dialog',
        title=trans._('Open File(s)...'),
        callback=QtViewer._open_files_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[StandardKeyBinding.Open],
    ),
    Action(
        id='napari.window.file.open_files_as_stack_dialog',
        title=trans._('Open Files as Stack...'),
        callback=QtViewer._open_files_dialog_as_stack_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyO}],
    ),
    Action(
        id='napari.window.file.open_folder_dialog',
        title=trans._('Open Folder...'),
        callback=QtViewer._open_files_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.NAVIGATION}],
        keybindings=[
            {'primary': KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyO}
        ],
    ),
    Action(
        id='napari.window.file._open_files_with_plugin',
        title=trans._('Open File(s)...'),
        callback=_open_files_with_plugin,
        menus=[
            {'id': MenuId.FILE_OPEN_WITH_PLUGIN, 'group': MenuGroup.NAVIGATION}
        ],
    ),
    Action(
        id='napari.window.file._open_files_as_stack_with_plugin',
        title=trans._('Open Files as Stack...'),
        callback=_open_files_as_stack_with_plugin,
        menus=[
            {'id': MenuId.FILE_OPEN_WITH_PLUGIN, 'group': MenuGroup.NAVIGATION}
        ],
    ),
    Action(
        id='napari.window.file._open_folder_with_plugin',
        title=trans._('Open Folder...'),
        callback=_open_folder_with_plugin,
        menus=[
            {'id': MenuId.FILE_OPEN_WITH_PLUGIN, 'group': MenuGroup.NAVIGATION}
        ],
    ),
    Action(
        id='napari.window.file.show_preferences_dialog',
        title=trans._('Preferences'),
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
        id='napari.window.file.save_layers_dialog.selected',
        title=trans._('Save Selected Layers...'),
        callback=_save_selected_layers,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[StandardKeyBinding.Save],
        enablement=(LLSCK.num_selected_layers > 0),
    ),
    Action(
        id='napari.window.file.save_layers_dialog',
        title=trans._('Save All Layers...'),
        callback=QtViewer._save_layers_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.CtrlCmd | KeyMod.Alt | KeyCode.KeyS}],
        enablement=(LLCK.num_layers > 0),
    ),
    Action(
        id='napari.window.file.save_canvas_screenshot_dialog',
        title=trans._('Save Screenshot...'),
        callback=QtViewer._screenshot_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyS}],
        status_tip=trans._(
            'Save screenshot of current display, default: .png'
        ),
    ),
    Action(
        id='napari.window.file.save_viewer_screenshot_dialog',
        title=trans._('Save Screenshot with Viewer...'),
        callback=Window._screenshot_dialog,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyS}],
        status_tip=trans._(
            'Save screenshot of current display, default: .png'
        ),
    ),
    Action(
        id='napari.window.file.copy_canvas_screenshot',
        title=trans._('Copy Screenshot to Clipboard'),
        callback=QtViewer.clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyC}],
        status_tip=trans._(
            'Copy screenshot of current display to the clipboard'
        ),
    ),
    Action(
        id='napari.window.file.copy_viewer_screenshot',
        title=trans._('Copy Screenshot with Viewer to Clipboard'),
        callback=Window.clipboard,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.SAVE}],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyC}],
        status_tip=trans._(
            'Copy screenshot of current display with the viewer to the clipboard'
        ),
    ),
    Action(
        id='napari.window.file.close_dialog',
        title=trans._('Close Window'),
        callback=_close_window,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.CLOSE}],
        keybindings=[StandardKeyBinding.Close],
    ),
    Action(
        id='napari.window.file.restart',
        title=trans._('Restart'),
        callback=_restart,
        menus=[
            {
                'id': MenuId.MENUBAR_FILE,
                'group': MenuGroup.CLOSE,
                'when': (
                    Path(sys.executable).parent / '.napari_is_bundled'
                ).exists(),
            }
        ],
    ),
    Action(
        id='napari.window.file.quit_dialog',
        title=trans._('Exit'),
        callback=_close_app,
        menus=[{'id': MenuId.MENUBAR_FILE, 'group': MenuGroup.CLOSE}],
        keybindings=[StandardKeyBinding.Quit],
    ),
]

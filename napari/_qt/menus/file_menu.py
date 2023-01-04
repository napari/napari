from itertools import chain
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize
from qtpy.QtWidgets import QAction, QMenu

from napari._qt.dialogs.preferences_dialog import PreferencesDialog
from napari._qt.dialogs.qt_reader_dialog import handle_gui_reading
from napari._qt.dialogs.screenshot_dialog import ScreenshotDialog
from napari._qt.menus._util import NapariMenu, populate_menu
from napari.errors.reader_errors import MultipleReaderError
from napari.settings import get_settings
from napari.utils.history import get_save_history, update_save_history
from napari.utils.misc import running_as_bundled_app
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari import Viewer
    from napari._qt.qt_main_window import Window


class FileMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&File'), window._qt_window)
        self.open_sample_menu = NapariMenu(trans._('Open Sample'), self)
        ACTIONS = [
            {
                'text': trans._('Open File(s)...'),
                'slot': window._qt_viewer._open_files_dialog,
                'shortcut': 'Ctrl+O',
            },
            {
                'text': trans._('Open Files as Stack...'),
                'slot': window._qt_viewer._open_files_dialog_as_stack_dialog,
                'shortcut': 'Ctrl+Alt+O',
            },
            {
                'text': trans._('Open Folder...'),
                'slot': window._qt_viewer._open_folder_dialog,
                'shortcut': 'Ctrl+Shift+O',
            },
            {
                'menu': trans._('Open with Plugin'),
                'items': [
                    {
                        'text': 'Open File(s)...',
                        'slot': self._open_files_w_plugin,
                    },
                    {
                        'text': 'Open Files as Stack...',
                        'slot': self._open_files_as_stack_w_plugin,
                    },
                    {
                        'text': 'Open Folder...',
                        'slot': self._open_folder_w_plugin,
                    },
                ],
            },
            {'menu': self.open_sample_menu},
            {},
            {
                'text': trans._('Preferences'),
                'slot': self._open_preferences,
                'shortcut': 'Ctrl+Shift+P',
                'statusTip': trans._('Open preferences dialog'),
                'menuRole': QAction.PreferencesRole,
            },
            {},
            {
                'text': trans._('Save Selected Layer(s)...'),
                'slot': lambda: window._qt_viewer._save_layers_dialog(
                    selected=True
                ),
                'shortcut': 'Ctrl+S',
                'enabled': self._layer_count,
            },
            {
                'text': trans._('Save All Layers...'),
                'slot': lambda: window._qt_viewer._save_layers_dialog(
                    selected=False
                ),
                'shortcut': 'Ctrl+Shift+S',
                'enabled': self._layer_count,
            },
            {
                'text': trans._('Save Screenshot...'),
                'slot': window._qt_viewer._screenshot_dialog,
                'shortcut': 'Alt+S',
                'statusTip': trans._(
                    'Save screenshot of current display, default .png'
                ),
            },
            {
                'text': trans._('Save Screenshot with Viewer...'),
                'slot': self._screenshot_dialog,
                'shortcut': 'Alt+Shift+S',
                'statusTip': trans._(
                    'Save screenshot of current display with the viewer, default .png'
                ),
            },
            {
                'text': trans._('Copy Screenshot to Clipboard'),
                'slot': window._qt_viewer.clipboard,
                'shortcut': 'Alt+C',
                'statusTip': trans._(
                    'Copy screenshot of current display to the clipboard'
                ),
            },
            {
                'text': trans._('Copy Screenshot with Viewer to Clipboard'),
                'slot': window.clipboard,
                'shortcut': 'Alt+Shift+C',
                'statusTip': trans._(
                    'Copy screenshot of current display with the viewer to the clipboard'
                ),
            },
            {},
            {
                'text': trans._('Close Window'),
                'slot': self._close_window,
                'shortcut': 'Ctrl+W',
            },
            {
                'when': running_as_bundled_app(),
                'text': trans._('Restart'),
                'slot': window._qt_window.restart,
            },
            # OS X will rename this to Quit and put it in the app menu.
            # This quits the entire QApplication and closes all windows.
            {
                'text': trans._('Exit'),
                'slot': self._close_app,
                'shortcut': 'Ctrl+Q',
                'menuRole': QAction.QuitRole,
            },
        ]
        populate_menu(self, ACTIONS)

        self._pref_dialog = None

        from napari.plugins import plugin_manager

        plugin_manager.discover_sample_data()
        plugin_manager.events.disabled.connect(self._rebuild_samples_menu)
        plugin_manager.events.registered.connect(self._rebuild_samples_menu)
        plugin_manager.events.unregistered.connect(self._rebuild_samples_menu)
        self._rebuild_samples_menu()
        self.update()

    def _close_app(self):
        self._win._qt_window.close(quit_app=True, confirm_need=True)

    def _close_window(self):
        self._win._qt_window.close(quit_app=False, confirm_need=True)

    def _layer_count(self, event=None):
        return len(self._win._qt_viewer.viewer.layers)

    def _screenshot_dialog(self):
        """Save screenshot of current display with viewer, default .png"""
        hist = get_save_history()
        dial = ScreenshotDialog(
            self._win.screenshot, self._win._qt_viewer, hist[0], hist
        )
        if dial.exec_():
            update_save_history(dial.selectedFiles()[0])

    def _open_preferences(self):
        """Edit preferences from the menubar."""
        if self._pref_dialog is None:
            win = PreferencesDialog(parent=self._win._qt_window)
            self._pref_dialog = win

            app_pref = get_settings().application
            if app_pref.preferences_size:
                win.resize(*app_pref.preferences_size)

            @win.resized.connect
            def _save_size(sz: QSize):
                app_pref.preferences_size = (sz.width(), sz.height())

            win.finished.connect(self._clean_pref_dialog)
            win.show()
        else:
            self._pref_dialog.raise_()

    def _clean_pref_dialog(self):
        self._pref_dialog = None

    def _rebuild_samples_menu(self):
        from napari.plugins import _npe2, menu_item_template, plugin_manager

        self.open_sample_menu.clear()

        for plugin_name, samples in chain(
            _npe2.sample_iterator(), plugin_manager._sample_data.items()
        ):
            multiprovider = len(samples) > 1
            if multiprovider:
                # use display_name for the menu item if npe2
                from npe2 import plugin_manager as pm

                try:
                    plugin_display_name = pm.get_manifest(
                        plugin_name
                    ).display_name
                except KeyError:
                    plugin_display_name = plugin_name
                menu = self.open_sample_menu.addMenu(
                    QMenu(title=plugin_display_name, parent=self)
                ).menu()
            else:
                menu = self.open_sample_menu

            for samp_name, samp_dict in samples.items():
                display_name = samp_dict['display_name'].replace("&", "&&")
                if multiprovider:
                    action = QAction(display_name, parent=self)
                else:
                    full_name = menu_item_template.format(
                        plugin_name, display_name
                    )
                    action = QAction(full_name, parent=self)

                def _add_sample(*args, plg=plugin_name, smp=samp_name):
                    try:
                        self._win._qt_viewer.viewer.open_sample(plg, smp)
                    except MultipleReaderError as e:
                        handle_gui_reading(
                            e.paths,
                            self._win._qt_viewer,
                            plugin_name=plugin_name,
                            stack=False,
                        )

                menu.addAction(action)
                action.triggered.connect(_add_sample)

    def _open_files_w_plugin(self):
        """Helper method for forcing plugin choice"""
        self._win._qt_viewer._open_files_dialog(choose_plugin=True)

    def _open_files_as_stack_w_plugin(self):
        """Helper method for forcing plugin choice"""
        self._win._qt_viewer._open_files_dialog_as_stack_dialog(
            choose_plugin=True
        )

    def _open_folder_w_plugin(self):
        """Helper method for forcing plugin choice"""
        self._win._qt_viewer._open_folder_dialog(choose_plugin=True)


def show_shortcuts(viewer: 'Viewer'):
    viewer.window.file_menu._open_preferences()
    pref_list = viewer.window.file_menu._pref_dialog._list
    for i in range(pref_list.count()):
        if pref_list.item(i).text() == "Shortcuts":
            pref_list.setCurrentRow(i)

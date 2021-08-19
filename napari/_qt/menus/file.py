from functools import partial
from typing import TYPE_CHECKING

from PyQt5.QtCore import QSize
from qtpy.QtWidgets import QAction, QMenu

from ...plugins import menu_item_template, plugin_manager
from ...settings import get_settings
from ...utils.history import get_save_history, update_save_history
from ...utils.misc import running_as_bundled_app
from ...utils.translations import trans
from ..dialogs.preferences_dialog import PreferencesDialog
from ..dialogs.screenshot_dialog import ScreenshotDialog

if TYPE_CHECKING:
    from ..qt_main_window import Window


class FileMenu(QMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(window._qt_window)
        self.setTitle(trans._('&File'))
        self.open_sample_menu = QMenu('Open Sample')
        ACTIONS = [
            {
                'text': trans._('Open File(s)...'),
                'slot': window.qt_viewer._open_files_dialog,
                'shortcut': 'Ctrl+O',
            },
            {
                'text': trans._('Open Files as Stack...'),
                'slot': window.qt_viewer._open_files_dialog_as_stack_dialog,
                'shortcut': 'Ctrl+Alt+O',
            },
            {
                'text': trans._('Open Folder...'),
                'slot': window.qt_viewer._open_folder_dialog,
                'shortcut': 'Ctrl+Shift+O',
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
                'slot': lambda: window.qt_viewer._save_layers_dialog(
                    selected=True
                ),
                'shortcut': 'Ctrl+S',
            },
            {
                'text': trans._('Save All Layers...'),
                'slot': lambda: window.qt_viewer._save_layers_dialog(
                    selected=False
                ),
                'shortcut': 'Ctrl+Shift+S',
            },
            {
                'text': trans._('Save Screenshot...'),
                'slot': window.qt_viewer._screenshot_dialog,
                'shortcut': 'Alt+S',
                'statusTip': 'Save screenshot of current display, default .png',
            },
            {
                'text': trans._('Save Screenshot with Viewer...'),
                'slot': self._screenshot_dialog,
                'shortcut': 'Alt+Shift+S',
                'statusTip': 'Save screenshot of current display with the viewer, default .png',
            },
            {
                'text': trans._('Copy Screenshot to Clipboard'),
                'slot': lambda: window.qt_viewer.clipboard(),
                'shortcut': 'Alt+Shift+S',
                'statusTip': 'Copy screenshot of current display to the clipboard',
            },
            {
                'text': trans._('Copy Screenshot with Viewer to Clipboard'),
                'slot': lambda: self.clipboard(),
                'shortcut': 'Alt+Shift+S',
                'statusTip': trans._(
                    'Copy screenshot of current display with the viewer to the clipboard'
                ),
            },
            {},
            {
                'text': trans._('Close Window'),
                'slot': window._qt_window.close_window,
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
                'slot': lambda: window._qt_window.close(quit_app=True),
                'shortcut': 'Ctrl+Q',
                'menuRole': QAction.QuitRole,
            },
        ]
        for ax in ACTIONS:
            if not ax:
                self.addSeparator()
                continue
            if not ax.get("when", True):
                continue
            if 'menu' in ax:
                self.addMenu(ax['menu'])
                continue
            action = self.addAction(ax['text'], ax['slot'])
            action.setShortcut(ax.get('shortcut', ''))
            action.setStatusTip(ax.get('statusTip', ''))
            if 'menuRole' in ax:
                action.setMenuRole(ax['menuRole'])

        self._pref_dialog = None
        self._pref_dialog_size = QSize()

        plugin_manager.discover_sample_data()
        plugin_manager.events.disabled.connect(self._rebuild_samples_menu)
        plugin_manager.events.registered.connect(self._rebuild_samples_menu)
        plugin_manager.events.unregistered.connect(self._rebuild_samples_menu)
        self._rebuild_samples_menu()

    def _screenshot_dialog(self):
        """Save screenshot of current display with viewer, default .png"""
        hist = get_save_history()
        dial = ScreenshotDialog(
            self.screenshot, self.window.qt_viewer, hist[0], hist
        )
        if dial.exec_():
            update_save_history(dial.selectedFiles()[0])

    def _open_preferences(self):
        """Edit preferences from the menubar."""
        if self._pref_dialog is None:
            win = PreferencesDialog(parent=self)
            # win.resized.connect(partial(setattr, self, '_pref_dialog_size'))
            win.valueChanged.connect(self._reset_preference_states)
            win.updatedValues.connect(self._win._update_widget_states)
            win.closed.connect(partial(setattr, self, '_pref_dialog', None))
            # if self._pref_dialog_size.isValid():
            # win.resize(self._pref_dialog_size)
            win.show()
            self._pref_dialog = win
        else:
            self._pref_dialog.raise_()

    def _rebuild_samples_menu(self, event=None):
        self.open_sample_menu.clear()

        for plugin_name, samples in plugin_manager._sample_data.items():
            multiprovider = len(samples) > 1
            if multiprovider:
                menu = self.open_sample_menu.addMenu(plugin_name)
            else:
                menu = self.open_sample_menu

            for samp_name, samp_dict in samples.items():
                display_name = samp_dict['display_name']
                if multiprovider:
                    action = QAction(display_name, parent=self)
                else:
                    full_name = menu_item_template.format(
                        plugin_name, display_name
                    )
                    action = QAction(full_name, parent=self)

                def _add_sample(*args, plg=plugin_name, smp=samp_name):
                    # window.qt_viewer.viewer.open_sample(plg, smp)
                    ...

                menu.addAction(action)
                action.triggered.connect(_add_sample)

    def clipboard(self, flash=True):
        """Take a screenshot of the currently displayed viewer and copy the image to the clipboard.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        from qtpy.QtGui import QGuiApplication

        img = self._screenshot(flash)
        cb = QGuiApplication.clipboard()
        cb.setImage(img)

    def _screenshot(self, flash=True):
        """Capture screenshot of the currently displayed viewer.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        img = self.grab().toImage()
        if flash:
            from ..utils import add_flash_animation

            add_flash_animation(self)
        return img

    def _reset_preference_states(self):
        # resetting plugin states in plugin manager
        plugin_manager.discover()

        # need to reset call order to defaults
        settings = get_settings()
        plugin_manager.set_call_order(
            settings.plugins.call_order
            or settings.plugins._defaults.get('call_order', {})
        )

        # reset the keybindings in action manager
        self._win.qt_viewer._bind_shortcuts()

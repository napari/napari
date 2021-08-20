"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
import inspect
import os
import sys
import time
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

from qtpy.QtCore import QEvent, QEventLoop, QPoint, QProcess, QSize, Qt, Slot
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QShortcut,
    QWidget,
)

from ..plugins import menu_item_template as plugin_menu_item_template
from ..plugins import plugin_manager
from ..settings import get_settings
from ..utils import config, perf
from ..utils.history import get_save_history, update_save_history
from ..utils.io import imsave
from ..utils.misc import in_jupyter, running_as_bundled_app
from ..utils.notifications import Notification
from ..utils.translations import trans
from .dialogs.activity_dialog import ActivityDialog, ActivityToggleItem
from .dialogs.preferences_dialog import PreferencesDialog
from .dialogs.qt_about import QtAbout
from .dialogs.qt_notification import NapariQtNotification
from .dialogs.qt_plugin_dialog import QtPluginDialog
from .dialogs.qt_plugin_report import QtPluginErrReporter
from .dialogs.screenshot_dialog import ScreenshotDialog
from .perf.qt_debug_menu import DebugMenu
from .qt_event_loop import NAPARI_ICON_PATH, get_app, quit_app
from .qt_resources import get_stylesheet
from .qt_viewer import QtViewer
from .utils import QImg2array, qbytearray_to_str, str_to_qbytearray
from .widgets.qt_viewer_dock_widget import (
    _SHORTCUT_DEPRECATION_STRING,
    QtViewerDockWidget,
)

_sentinel = object()


class _QtMainWindow(QMainWindow):
    # This was added so that someone can patch
    # `napari._qt.qt_main_window._QtMainWindow._window_icon`
    # to their desired window icon
    _window_icon = NAPARI_ICON_PATH

    # To track window instances and facilitate getting the "active" viewer...
    # We use this instead of QApplication.activeWindow for compatibility with
    # IPython usage. When you activate IPython, it will appear that there are
    # *no* active windows, so we want to track the most recently active windows
    _instances: ClassVar[List['_QtMainWindow']] = []

    def __init__(self, qt_viewer: QtViewer, parent=None) -> None:
        super().__init__(parent)
        self._ev = None
        self.qt_viewer = qt_viewer

        self._quit_app = False
        self.setWindowIcon(QIcon(self._window_icon))
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setUnifiedTitleAndToolBarOnMac(True)
        center = QWidget(self)
        center.setLayout(QHBoxLayout())
        center.layout().addWidget(qt_viewer)
        center.layout().setContentsMargins(4, 0, 4, 0)
        self.setCentralWidget(center)

        self.setWindowTitle(qt_viewer.viewer.title)

        self._maximized_flag = False
        self._preferences_dialog = None
        self._preferences_dialog_size = QSize()
        self._window_size = None
        self._window_pos = None
        self._old_size = None
        self._positions = []

        self._activity_dialog = ActivityDialog()
        self._status_bar = self.statusBar()

        settings = get_settings()

        # TODO:
        # settings.plugins.defaults.call_order = plugin_manager.call_order()

        # set the values in plugins to match the ones saved in settings
        if settings.plugins.call_order is not None:
            plugin_manager.set_call_order(settings.plugins.call_order)

        _QtMainWindow._instances.append(self)
        self.qt_viewer.viewer.tooltip.events.text.connect(self.update_tooltip)

    def update_tooltip(self, event):
        if self.qt_viewer.viewer.tooltip.visible:
            self.qt_viewer.setToolTip(event.value)
        else:
            self.qt_viewer.setToolTip("")

        # Connect the notification dispacther to correctly propagate
        # notifications from threads. See: `napari._qt.qt_event_loop::get_app`
        application_instance = QApplication.instance()
        if application_instance:
            application_instance._dispatcher.sig_notified.connect(
                self.show_notification
            )

    @classmethod
    def current(cls):
        return cls._instances[-1] if cls._instances else None

    @classmethod
    def current_viewer(cls):
        window = cls.current()
        return window.qt_viewer.viewer if window else None

    def event(self, e):
        if e.type() == QEvent.Close:
            # when we close the MainWindow, remove it from the instances list
            try:
                _QtMainWindow._instances.remove(self)
            except ValueError:
                pass
        if e.type() in {QEvent.WindowActivate, QEvent.ZOrderChange}:
            # upon activation or raise_, put window at the end of _instances
            try:
                inst = _QtMainWindow._instances
                inst.append(inst.pop(inst.index(self)))
            except ValueError:
                pass
        return super().event(e)

    def _load_window_settings(self):
        """
        Load window layout settings from configuration.
        """
        settings = get_settings()
        window_position = settings.application.window_position

        # It's necessary to verify if the window/position value is valid with
        # the current screen.
        if not window_position:
            window_position = (self.x(), self.y())
        else:
            width, height = window_position
            screen_geo = QApplication.desktop().geometry()
            if screen_geo.width() < width or screen_geo.height() < height:
                window_position = (self.x(), self.y())

        return (
            settings.application.window_state,
            settings.application.window_size,
            window_position,
            settings.application.window_maximized,
            settings.application.window_fullscreen,
            settings.application.preferences_size,
        )

    def _get_window_settings(self):
        """Return current window settings.

        Symmetric to the 'set_window_settings' setter.
        """

        window_fullscreen = self.isFullScreen()
        if window_fullscreen:
            window_maximized = self._maximized_flag
        else:
            window_maximized = self.isMaximized()

        preferences_dialog_size = (
            self._preferences_dialog_size.width(),
            self._preferences_dialog_size.height(),
        )
        window_state = qbytearray_to_str(self.saveState())
        return (
            window_state,
            self._window_size or (self.width(), self.height()),
            self._window_pos or (self.x(), self.y()),
            window_maximized,
            window_fullscreen,
            preferences_dialog_size,
        )

    def _set_window_settings(
        self,
        window_state,
        window_size,
        window_position,
        window_maximized,
        window_fullscreen,
        preferences_dialog_size,
    ):
        """
        Set window settings.

        Symmetric to the 'get_window_settings' accessor.
        """
        self.setUpdatesEnabled(False)
        self.setWindowState(Qt.WindowNoState)

        if preferences_dialog_size:
            self._preferences_dialog_size = QSize(*preferences_dialog_size)

        if window_position:
            window_position = QPoint(*window_position)
            self.move(window_position)

        if window_size:
            window_size = QSize(*window_size)
            self.resize(window_size)

        if window_state:
            self.restoreState(str_to_qbytearray(window_state))

        if window_fullscreen:
            self.setWindowState(Qt.WindowFullScreen)
            self._maximized_flag = window_maximized
        elif window_maximized:
            self.setWindowState(Qt.WindowMaximized)

        self.setUpdatesEnabled(True)

    def _save_current_window_settings(self):
        """Save the current geometry of the main window."""
        (
            window_state,
            window_size,
            window_position,
            window_maximized,
            window_fullscreen,
            preferences_dialog_size,
        ) = self._get_window_settings()

        settings = get_settings()
        if settings.application.save_window_geometry:
            settings.application.window_maximized = window_maximized
            settings.application.window_fullscreen = window_fullscreen
            settings.application.window_position = window_position
            settings.application.window_size = window_size
            settings.application.window_statusbar = (
                not self._status_bar.isHidden()
            )
            settings.application.preferences_size = preferences_dialog_size

        if settings.application.save_window_state:
            settings.application.window_state = window_state

    def close(self, quit_app=False):
        """Override to handle closing app or just the window."""
        self._quit_app = quit_app
        return super().close()

    def close_window(self):
        """Close active dialog or active window."""
        parent = QApplication.focusWidget()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                self.close()
                break

            if isinstance(parent, QDialog):
                parent.close()
                break

            try:
                parent = parent.parent()
            except Exception:
                parent = getattr(parent, "_parent", None)

    def show(self, block=False):
        super().show()
        if block:
            self._ev = QEventLoop()
            self._ev.exec()

    def changeEvent(self, event):
        """Handle window state changes."""
        if event.type() == QEvent.WindowStateChange:
            # TODO: handle maximization issue. When double clicking on the
            # title bar on Mac the resizeEvent is called an varying amount
            # of times which makes it hard to track the original size before
            # maximization.
            condition = (
                self.isMaximized() if os.name == "nt" else self.isFullScreen()
            )
            if condition:
                if self._positions and len(self._positions) > 1:
                    self._window_pos = self._positions[-2]

                self._window_size = (
                    self._old_size.width(),
                    self._old_size.height(),
                )
            else:
                self._old_size = None
                self._window_pos = None
                self._window_size = None
                self._positions = []

        super().changeEvent(event)

    def resizeEvent(self, event):
        """Override to handle original size before maximizing."""
        # the first resize event will have nonsense positions that we dont
        # want to store (and potentially restore)
        if event.oldSize().isValid():
            self._old_size = event.oldSize()
            self._positions.append((self.x(), self.y()))

            if self._positions and len(self._positions) >= 2:
                self._window_pos = self._positions[-2]
                self._positions = self._positions[-2:]

        super().resizeEvent(event)

    def closeEvent(self, event):
        """This method will be called when the main window is closing.

        Regardless of whether cmd Q, cmd W, or the close button is used...
        """
        if self._ev and self._ev.isRunning():
            self._ev.quit()

        # Close any floating dockwidgets
        for dock in self.findChildren(QtViewerDockWidget):
            if dock.isFloating():
                dock.setFloating(False)

        self._save_current_window_settings()

        # On some versions of Darwin, exiting while fullscreen seems to tickle
        # some bug deep in NSWindow.  This forces the fullscreen keybinding
        # test to complete its draw cycle, then pop back out of fullscreen.
        if self.isFullScreen():
            self.showNormal()
            for _i in range(5):
                time.sleep(0.1)
                QApplication.processEvents()

        if self._quit_app:
            quit_app()

        event.accept()

    def restart(self):
        """Restart the napari application in a detached process."""
        process = QProcess()
        process.setProgram(sys.executable)

        if not running_as_bundled_app():
            process.setArguments(sys.argv)

        process.startDetached()
        self.close(quit_app=True)

    @staticmethod
    @Slot(Notification)
    def show_notification(notification: Notification):
        """Show notification coming from a thread."""
        NapariQtNotification.show_notification(notification)


class Window:
    """Application window that contains the menu bar and viewer.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Contained viewer widget.

    Attributes
    ----------
    file_menu : qtpy.QtWidgets.QMenu
        File menu.
    help_menu : qtpy.QtWidgets.QMenu
        Help menu.
    main_menu : qtpy.QtWidgets.QMainWindow.menuBar
        Main menubar.
    qt_viewer : QtViewer
        Contained viewer widget.
    view_menu : qtpy.QtWidgets.QMenu
        View menu.
    window_menu : qtpy.QtWidgets.QMenu
        Window menu.
    """

    def __init__(self, viewer, *, show: bool = True):
        settings = get_settings()

        # create QApplication if it doesn't already exist
        get_app()

        self._unnamed_dockwidget_count = 1

        # Connect the Viewer and create the Main Window
        self.qt_viewer = QtViewer(viewer, show_welcome_screen=True)
        self._qt_window = _QtMainWindow(self.qt_viewer)
        self._status_bar = self._qt_window.statusBar()

        # Dictionary holding dock widgets
        self._dock_widgets: Dict[str, QtViewerDockWidget] = {}

        # since we initialize canvas before window, we need to manually connect them again.
        if self._qt_window.windowHandle() is not None:
            self._qt_window.windowHandle().screenChanged.connect(
                self.qt_viewer.canvas._backend.screen_changed
            )

        self._add_menubar()
        self._add_file_menu()
        self._add_view_menu()
        self._add_window_menu()
        self._add_plugins_menu()
        self._add_help_menu()

        self._status_bar.showMessage(trans._('Ready'))
        self._help = QLabel('')
        self._status_bar.addPermanentWidget(self._help)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        self._qt_window._activity_dialog._toggleButton = self._activity_item

        canvas_widg = self.qt_viewer._canvas_overlay
        self._qt_window._activity_dialog.setParent(canvas_widg)
        self.qt_viewer._canvas_overlay.resized.connect(
            self._qt_window._activity_dialog.move_to_bottom_right
        )
        self._qt_window._activity_dialog.move_to_bottom_right()
        self._qt_window._activity_dialog.hide()
        self._status_bar.addPermanentWidget(self._activity_item)

        self.qt_viewer.viewer.theme = settings.appearance.theme
        self._update_theme()

        self._add_viewer_dock_widget(self.qt_viewer.dockConsole, tabify=False)
        self._add_viewer_dock_widget(
            self.qt_viewer.dockLayerControls, tabify=False
        )
        self._add_viewer_dock_widget(
            self.qt_viewer.dockLayerList, tabify=False
        )
        self.window_menu.addSeparator()

        settings.appearance.events.theme.connect(self._update_theme)

        plugin_manager.events.disabled.connect(self._rebuild_plugins_menu)
        plugin_manager.events.disabled.connect(self._rebuild_samples_menu)
        plugin_manager.events.registered.connect(self._rebuild_plugins_menu)
        plugin_manager.events.registered.connect(self._rebuild_samples_menu)
        plugin_manager.events.unregistered.connect(self._rebuild_plugins_menu)
        plugin_manager.events.unregistered.connect(self._rebuild_samples_menu)

        viewer.events.status.connect(self._status_changed)
        viewer.events.help.connect(self._help_changed)
        viewer.events.title.connect(self._title_changed)
        viewer.events.theme.connect(self._update_theme)

        if perf.USE_PERFMON:
            # Add DebugMenu and dockPerformance if using perfmon.
            self._debug_menu = DebugMenu(self)
            self._add_viewer_dock_widget(self.qt_viewer.dockPerformance)
        else:
            self._debug_menu = None

        if show:
            self.show()

    def _add_menubar(self):
        """Add menubar to napari app."""
        self.main_menu = self._qt_window.menuBar()
        # Menubar shortcuts are only active when the menubar is visible.
        # Therefore, we set a global shortcut not associated with the menubar
        # to toggle visibility, *but*, in order to not shadow the menubar
        # shortcut, we disable it, and only enable it when the menubar is
        # hidden. See this stackoverflow link for details:
        # https://stackoverflow.com/questions/50537642/how-to-keep-the-shortcuts-of-a-hidden-widget-in-pyqt5
        self._main_menu_shortcut = QShortcut(
            QKeySequence('Ctrl+M'), self._qt_window
        )
        self._main_menu_shortcut.activated.connect(
            self._toggle_menubar_visible
        )
        self._main_menu_shortcut.setEnabled(False)

    def _toggle_menubar_visible(self):
        """Toggle visibility of app menubar.

        This function also disables or enables a global keyboard shortcut to
        show the menubar, since menubar shortcuts are only available while the
        menubar is visible.
        """
        if self.main_menu.isVisible():
            self.main_menu.setVisible(False)
            self._main_menu_shortcut.setEnabled(True)
        else:
            self.main_menu.setVisible(True)
            self._main_menu_shortcut.setEnabled(False)

    def _add_file_menu(self):
        """Add 'File' menu to app menubar."""
        open_images = QAction(trans._('Open File(s)...'), self._qt_window)
        open_images.setShortcut('Ctrl+O')
        open_images.setStatusTip(trans._('Open file(s)'))
        open_images.triggered.connect(self.qt_viewer._open_files_dialog)

        open_stack = QAction(
            trans._('Open Files as Stack...'), self._qt_window
        )
        open_stack.setShortcut('Ctrl+Alt+O')
        open_stack.setStatusTip(trans._('Open files'))
        open_stack.triggered.connect(
            self.qt_viewer._open_files_dialog_as_stack_dialog
        )

        open_folder = QAction(trans._('Open Folder...'), self._qt_window)
        open_folder.setShortcut('Ctrl+Shift+O')
        open_folder.setStatusTip(trans._('Open a folder'))
        open_folder.triggered.connect(self.qt_viewer._open_folder_dialog)

        # OS X will rename this to Quit and put it in the app menu.
        preferences = QAction(trans._('Preferences'), self._qt_window)
        preferences.setShortcut('Ctrl+Shift+P')
        preferences.setStatusTip(trans._('Open preferences dialog'))
        preferences.setMenuRole(QAction.PreferencesRole)
        preferences.triggered.connect(self._open_preferences)

        save_selected_layers = QAction(
            trans._('Save Selected Layer(s)...'), self._qt_window
        )
        save_selected_layers.setShortcut('Ctrl+S')
        save_selected_layers.setStatusTip(trans._('Save selected layers'))
        save_selected_layers.triggered.connect(
            lambda: self.qt_viewer._save_layers_dialog(selected=True)
        )

        save_all_layers = QAction(
            trans._('Save All Layers...'), self._qt_window
        )
        save_all_layers.setShortcut('Ctrl+Shift+S')
        save_all_layers.setStatusTip(trans._('Save all layers'))
        save_all_layers.triggered.connect(
            lambda: self.qt_viewer._save_layers_dialog(selected=False)
        )

        screenshot = QAction(trans._('Save Screenshot...'), self._qt_window)
        screenshot.setShortcut('Alt+S')
        screenshot.setStatusTip(
            trans._('Save screenshot of current display, default .png')
        )
        screenshot.triggered.connect(self.qt_viewer._screenshot_dialog)

        screenshot_wv = QAction(
            trans._('Save Screenshot with Viewer...'), self._qt_window
        )
        screenshot_wv.setShortcut('Alt+Shift+S')
        screenshot_wv.setStatusTip(
            trans._(
                'Save screenshot of current display with the viewer, default .png'
            )
        )
        screenshot_wv.triggered.connect(self._screenshot_dialog)

        clipboard = QAction(
            trans._('Copy Screenshot to Clipboard'), self._qt_window
        )
        clipboard.setStatusTip(
            trans._('Copy screenshot of current display to the clipboard')
        )
        clipboard.triggered.connect(lambda: self.qt_viewer.clipboard())

        clipboard_wv = QAction(
            trans._('Copy Screenshot with Viewer to Clipboard'),
            self._qt_window,
        )
        clipboard_wv.setStatusTip(
            trans._(
                'Copy screenshot of current display with the viewer to the clipboard'
            )
        )
        clipboard_wv.triggered.connect(lambda: self.clipboard())

        # OS X will rename this to Quit and put it in the app menu.
        # This quits the entire QApplication and all windows that may be open.
        quitAction = QAction(trans._('Exit'), self._qt_window)
        quitAction.setShortcut('Ctrl+Q')
        quitAction.setMenuRole(QAction.QuitRole)
        quitAction.triggered.connect(
            lambda: self._qt_window.close(quit_app=True)
        )

        if running_as_bundled_app():
            restartAction = QAction(trans._('Restart'), self._qt_window)
            restartAction.triggered.connect(self._qt_window.restart)

        closeAction = QAction(trans._('Close Window'), self._qt_window)
        closeAction.setShortcut('Ctrl+W')
        closeAction.triggered.connect(self._qt_window.close_window)

        plugin_manager.discover_sample_data()
        self.open_sample_menu = QMenu(trans._('Open Sample'), self._qt_window)

        self._rebuild_samples_menu()

        self.file_menu = self.main_menu.addMenu(trans._('&File'))
        self.file_menu.addAction(open_images)
        self.file_menu.addAction(open_stack)
        self.file_menu.addAction(open_folder)
        self.file_menu.addMenu(self.open_sample_menu)
        self.file_menu.addSeparator()
        self.file_menu.addAction(preferences)
        self.file_menu.addSeparator()
        self.file_menu.addAction(save_selected_layers)
        self.file_menu.addAction(save_all_layers)
        self.file_menu.addAction(screenshot)
        self.file_menu.addAction(screenshot_wv)
        self.file_menu.addAction(clipboard)
        self.file_menu.addAction(clipboard_wv)
        self.file_menu.addSeparator()
        self.file_menu.addAction(closeAction)

        if running_as_bundled_app():
            self.file_menu.addAction(restartAction)

        self.file_menu.addAction(quitAction)

    def _rebuild_samples_menu(self, event=None):
        self.open_sample_menu.clear()

        for plugin_name, samples in plugin_manager._sample_data.items():
            multiprovider = len(samples) > 1
            if multiprovider:
                menu = QMenu(plugin_name, self._qt_window)
                self.open_sample_menu.addMenu(menu)
            else:
                menu = self.open_sample_menu

            for samp_name, samp_dict in samples.items():
                display_name = samp_dict['display_name']
                if multiprovider:
                    action = QAction(display_name, parent=self._qt_window)
                else:
                    full_name = plugin_menu_item_template.format(
                        plugin_name, display_name
                    )
                    action = QAction(full_name, parent=self._qt_window)

                def _add_sample(*args, plg=plugin_name, smp=samp_name):
                    self.qt_viewer.viewer.open_sample(plg, smp)

                menu.addAction(action)
                action.triggered.connect(_add_sample)

    def _open_preferences(self):
        """Edit preferences from the menubar."""
        if self._qt_window._preferences_dialog is None:
            win = PreferencesDialog(parent=self._qt_window)
            self._qt_window._preferences_dialog = win
            if self._qt_window._preferences_dialog_size:
                win.resize(self._qt_window._preferences_dialog_size)
            win.resized.connect(
                lambda e: setattr(
                    self._qt_window, '_preferences_dialog_size', e
                )
            )
            win.finished.connect(
                lambda e: setattr(self._qt_window, '_preferences_dialog', None)
            )
            win.show()
        else:
            self._qt_window._preferences_dialog.raise_()

    def _add_view_menu(self):
        """Add 'View' menu to app menubar."""
        settings = get_settings()
        toggle_visible = QAction(
            trans._('Toggle Menubar Visibility'), self._qt_window
        )
        toggle_visible.setShortcut('Ctrl+M')
        toggle_visible.setStatusTip(trans._('Hide Menubar'))
        toggle_visible.triggered.connect(self._toggle_menubar_visible)
        toggle_fullscreen = QAction(
            trans._('Toggle Full Screen'), self._qt_window
        )
        toggle_fullscreen.setShortcut('Ctrl+F')
        toggle_fullscreen.setStatusTip(trans._('Toggle full screen'))
        toggle_fullscreen.triggered.connect(self._toggle_fullscreen)
        toggle_play = QAction(trans._('Toggle Play'), self._qt_window)
        toggle_play.triggered.connect(self._toggle_play)
        toggle_play.setShortcut('Ctrl+Alt+P')
        toggle_play.setStatusTip(trans._('Toggle Play'))

        self.view_menu = self.main_menu.addMenu(trans._('&View'))
        self.view_menu.addAction(toggle_fullscreen)
        self.view_menu.addAction(toggle_visible)
        self.view_menu.addAction(toggle_play)
        self.view_menu.addSeparator()

        # Add octree actions.
        if config.async_octree:
            toggle_outline = QAction(
                trans._('Toggle Chunk Outlines'), self._qt_window
            )
            toggle_outline.triggered.connect(
                self.qt_viewer._toggle_chunk_outlines
            )
            toggle_outline.setShortcut('Ctrl+Alt+O')
            toggle_outline.setStatusTip(trans._('Toggle Chunk Outlines'))
            self.view_menu.addAction(toggle_outline)

        # Add axes menu
        axes = self.qt_viewer.viewer.axes
        axes_menu = QMenu(trans._('Axes'), parent=self._qt_window)
        axes_visible_action = QAction(
            trans._('Visible'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.axes.visible,
        )
        axes_visible_action.triggered.connect(self._toggle_axes_visible)
        self._event_to_action(axes_visible_action, axes.events.visible)
        axes_colored_action = QAction(
            trans._('Colored'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.axes.colored,
        )
        axes_colored_action.triggered.connect(self._toggle_axes_colored)
        self._event_to_action(axes_colored_action, axes.events.colored)
        axes_labels_action = QAction(
            trans._('Labels'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.axes.labels,
        )
        axes_labels_action.triggered.connect(self._toggle_axes_labels)
        self._event_to_action(axes_labels_action, axes.events.labels)
        axes_dashed_action = QAction(
            trans._('Dashed'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.axes.dashed,
        )
        axes_dashed_action.triggered.connect(self._toggle_axes_dashed)
        self._event_to_action(axes_dashed_action, axes.events.dashed)
        axes_arrows_action = QAction(
            trans._('Arrows'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.axes.arrows,
        )
        axes_arrows_action.triggered.connect(self._toggle_axes_arrows)
        self._event_to_action(axes_arrows_action, axes.events.arrows)

        axes_menu.addAction(axes_visible_action)
        axes_menu.addAction(axes_colored_action)
        axes_menu.addAction(axes_labels_action)
        axes_menu.addAction(axes_dashed_action)
        axes_menu.addAction(axes_arrows_action)
        self.view_menu.addMenu(axes_menu)

        # Add scale bar menu
        scale_bar = self.qt_viewer.viewer.scale_bar
        scale_bar_menu = QMenu(trans._('Scale Bar'), parent=self._qt_window)
        scale_bar_visible_action = QAction(
            trans._('Visible'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.scale_bar.visible,
        )
        scale_bar_visible_action.triggered.connect(
            self._toggle_scale_bar_visible
        )
        self._event_to_action(
            scale_bar_visible_action, scale_bar.events.visible
        )
        scale_bar_colored_action = QAction(
            trans._('Colored'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.scale_bar.colored,
        )
        scale_bar_colored_action.triggered.connect(
            self._toggle_scale_bar_colored
        )
        self._event_to_action(
            scale_bar_colored_action, scale_bar.events.colored
        )
        scale_bar_ticks_action = QAction(
            trans._('Ticks'),
            parent=self._qt_window,
            checkable=True,
            checked=self.qt_viewer.viewer.scale_bar.ticks,
        )
        scale_bar_ticks_action.triggered.connect(self._toggle_scale_bar_ticks)
        self._event_to_action(scale_bar_ticks_action, scale_bar.events.ticks)

        scale_bar_menu.addAction(scale_bar_visible_action)
        scale_bar_menu.addAction(scale_bar_colored_action)
        scale_bar_menu.addAction(scale_bar_ticks_action)
        self.view_menu.addMenu(scale_bar_menu)
        self.tooltip_menu = QAction(
            trans._('Layer Tooltip visibility'),
            parent=self._qt_window,
            checkable=True,
            checked=settings.appearance.layer_tooltip_visibility,
        )
        self.tooltip_menu.triggered.connect(self._tooltip_visibility_toggle)
        settings.appearance.events.layer_tooltip_visibility.connect(
            self._tooltip_visibility_toggled
        )
        self.view_menu.addAction(self.tooltip_menu)

        self.view_activity_menu = QAction(
            trans._('Activity Dock'),
            parent=self._qt_window,
            checkable=True,
            checked=self._qt_window._activity_dialog.isVisible(),
        )
        self.view_activity_menu.triggered.connect(self._toggle_activity_dock)
        self.view_menu.addAction(self.view_activity_menu)

        self.view_menu.addSeparator()

    def _tooltip_visibility_toggle(self, value):
        get_settings().appearance.layer_tooltip_visibility = value

    def _tooltip_visibility_toggled(self, event):
        self.tooltip_menu.setChecked(
            get_settings().appearance.layer_tooltip_visibility
        )

    def _event_to_action(self, action, event):
        """Connect triggered event in model to respective action in menu."""
        # TODO: use action manager to keep in sync
        event.connect(lambda e: action.setChecked(e.value))

    def _add_window_menu(self):
        """Add 'Window' menu to app menubar."""
        clear_action = QAction(trans._("Remove Dock Widgets"), self._qt_window)
        clear_action.setStatusTip(trans._('Remove all dock widgets'))
        clear_action.triggered.connect(
            lambda e: self.remove_dock_widget('all')
        )

        self.window_menu = self.main_menu.addMenu(trans._('&Window'))
        self.window_menu.addAction(clear_action)
        self.window_menu.addSeparator()

    def _add_plugins_menu(self):
        """Add 'Plugins' menu to app menubar."""
        self.plugins_menu = self.main_menu.addMenu(trans._('&Plugins'))

        plugin_manager.discover_widgets()
        self._rebuild_plugins_menu()

    def _rebuild_plugins_menu(self, event=None):

        self.plugins_menu.clear()

        pip_install_action = QAction(
            trans._("Install/Uninstall Plugins..."), self._qt_window
        )
        pip_install_action.triggered.connect(self._show_plugin_install_dialog)
        self.plugins_menu.addAction(pip_install_action)

        report_plugin_action = QAction(
            trans._("Plugin Errors..."), self._qt_window
        )
        report_plugin_action.setStatusTip(
            trans._(
                'Review stack traces for plugin exceptions and notify developers'
            )
        )
        report_plugin_action.triggered.connect(self._show_plugin_err_reporter)

        self.plugins_menu.addAction(report_plugin_action)

        self.plugins_menu.addSeparator()

        # Add a menu item (QAction) for each available plugin widget
        for hook_type, (plugin_name, widgets) in plugin_manager.iter_widgets():
            multiprovider = len(widgets) > 1
            if multiprovider:
                menu = QMenu(plugin_name, self._qt_window)
                self.plugins_menu.addMenu(menu)
            else:
                menu = self.plugins_menu

            for wdg_name in widgets:
                key = (plugin_name, wdg_name)
                if multiprovider:
                    action = QAction(wdg_name, parent=self._qt_window)
                else:
                    full_name = plugin_menu_item_template.format(*key)
                    action = QAction(full_name, parent=self._qt_window)

                def _add_widget(*args, key=key, hook_type=hook_type):
                    if hook_type == 'dock':
                        self.add_plugin_dock_widget(*key)
                    else:
                        self._add_plugin_function_widget(*key)

                menu.addAction(action)
                action.triggered.connect(_add_widget)

    def _show_plugin_install_dialog(self):
        """Show dialog that allows users to sort the call order of plugins."""

        self.plugin_dialog = QtPluginDialog(self._qt_window)
        self.plugin_dialog.exec_()

    def _show_plugin_err_reporter(self):
        """Show dialog that allows users to review and report plugin errors."""
        QtPluginErrReporter(parent=self._qt_window).exec_()

    def _add_help_menu(self):
        """Add 'Help' menu to app menubar."""
        self.help_menu = self.main_menu.addMenu(trans._('&Help'))

        about_action = QAction(trans._("napari Info"), self._qt_window)
        about_action.setShortcut("Ctrl+/")
        about_action.setStatusTip(trans._('About napari'))
        about_action.triggered.connect(
            lambda e: QtAbout.showAbout(self.qt_viewer, self._qt_window)
        )
        self.help_menu.addAction(about_action)

    def _toggle_activity_dock(self, event):
        is_currently_visible = self._qt_window._activity_dialog.isVisible()
        if not is_currently_visible:
            self._qt_window._activity_dialog.show()
            self._qt_window._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(Qt.DownArrow)
            self.view_activity_menu.setChecked(True)
        else:
            self._qt_window._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.UpArrow)
            self.view_activity_menu.setChecked(False)

    def _toggle_scale_bar_visible(self, state):
        self.qt_viewer.viewer.scale_bar.visible = state

    def _toggle_scale_bar_colored(self, state):
        self.qt_viewer.viewer.scale_bar.colored = state

    def _toggle_scale_bar_ticks(self, state):
        self.qt_viewer.viewer.scale_bar.ticks = state

    def _toggle_axes_visible(self, state):
        self.qt_viewer.viewer.axes.visible = state

    def _toggle_axes_colored(self, state):
        self.qt_viewer.viewer.axes.colored = state

    def _toggle_axes_labels(self, state):
        self.qt_viewer.viewer.axes.labels = state

    def _toggle_axes_dashed(self, state):
        self.qt_viewer.viewer.axes.dashed = state

    def _toggle_axes_arrows(self, state):
        self.qt_viewer.viewer.axes.arrows = state

    def _toggle_fullscreen(self, event):
        """Toggle fullscreen mode."""
        if self._qt_window.isFullScreen():
            self._qt_window.showNormal()
        else:
            self._qt_window.showFullScreen()

    def _toggle_play(self, state):
        """Toggle play."""
        if self.qt_viewer.dims.is_playing:
            self.qt_viewer.dims.stop()
        else:
            axis = self.qt_viewer.viewer.dims.last_used or 0
            self.qt_viewer.dims.play(axis)

    def add_plugin_dock_widget(
        self, plugin_name: str, widget_name: str = None
    ) -> Tuple[QtViewerDockWidget, Any]:
        """Add plugin dock widget if not already added.

        Parameters
        ----------
        plugin_name : str
            Name of a plugin providing a widget
        widget_name : str, optional
            Name of a widget provided by `plugin_name`. If `None`, and the
            specified plugin provides only a single widget, that widget will be
            returned, otherwise a ValueError will be raised, by default None

        Returns
        -------
        tuple
            A 2-tuple containing (the DockWidget instance, the plugin widget
            instance).
        """
        from ..viewer import Viewer

        Widget, dock_kwargs = plugin_manager.get_widget(
            plugin_name, widget_name
        )
        if not widget_name:
            # if widget_name wasn't provided, `get_widget` will have
            # ensured that there is a single widget available.
            widget_name = list(plugin_manager._dock_widgets[plugin_name])[0]

        full_name = plugin_menu_item_template.format(plugin_name, widget_name)
        if full_name in self._dock_widgets:
            dock_widget = self._dock_widgets[full_name]
            dock_widget.show()
            wdg = dock_widget.widget()
            if hasattr(wdg, '_magic_widget'):
                wdg = wdg._magic_widget
            return dock_widget, wdg

        # if the signature is looking a for a napari viewer, pass it.
        kwargs = {}
        for param in inspect.signature(Widget.__init__).parameters.values():
            if param.name == 'napari_viewer':
                kwargs['napari_viewer'] = self.qt_viewer.viewer
                break
            if param.annotation in ('napari.viewer.Viewer', Viewer):
                kwargs[param.name] = self.qt_viewer.viewer
                break
            # cannot look for param.kind == param.VAR_KEYWORD because
            # QWidget allows **kwargs but errs on unknown keyword arguments

        # instantiate the widget
        wdg = Widget(**kwargs)

        # Add dock widget
        dock_kwargs.pop('name', None)
        dock_widget = self.add_dock_widget(wdg, name=full_name, **dock_kwargs)
        return dock_widget, wdg

    def _add_plugin_function_widget(self, plugin_name: str, widget_name: str):
        """Add plugin function widget if not already added.

        Parameters
        ----------
        plugin_name : str
            Name of a plugin providing a widget
        widget_name : str, optional
            Name of a widget provided by `plugin_name`. If `None`, and the
            specified plugin provides only a single widget, that widget will be
            returned, otherwise a ValueError will be raised, by default None
        """
        full_name = plugin_menu_item_template.format(plugin_name, widget_name)
        if full_name in self._dock_widgets:
            self._dock_widgets[full_name].show()
            return

        func = plugin_manager._function_widgets[plugin_name][widget_name]

        # Add function widget
        self.add_function_widget(
            func, name=full_name, area=None, allowed_areas=None
        )

    def add_dock_widget(
        self,
        widget: QWidget,
        *,
        name: str = '',
        area: str = 'right',
        allowed_areas: Optional[Sequence[str]] = None,
        shortcut=_sentinel,
        add_vertical_stretch=True,
    ):
        """Convenience method to add a QDockWidget to the main window.

        If name is not provided a generic name will be addded to avoid
        `saveState` warnings on close.

        Parameters
        ----------
        widget : QWidget
            `widget` will be added as QDockWidget's main widget.
        name : str, optional
            Name of dock widget to appear in window menu.
        area : str
            Side of the main window to which the new dock widget will be added.
            Must be in {'left', 'right', 'top', 'bottom'}
        allowed_areas : list[str], optional
            Areas, relative to main window, that the widget is allowed dock.
            Each item in list must be in {'left', 'right', 'top', 'bottom'}
            By default, all areas are allowed.
        shortcut : str, optional
            Keyboard shortcut to appear in dropdown menu.
        add_vertical_stretch : bool, optional
            Whether to add stretch to the bottom of vertical widgets (pushing
            widgets up towards the top of the allotted area, instead of letting
            them distribute across the vertical space).  By default, True.

            .. deprecated:: 0.4.8

                The shortcut parameter is deprecated since version 0.4.8, please use
                the action and shortcut manager APIs. The new action manager and
                shortcut API allow user configuration and localisation.

        Returns
        -------
        dock_widget : QtViewerDockWidget
            `dock_widget` that can pass viewer events.
        """
        if not name:
            try:
                name = widget.objectName()
            except AttributeError:
                name = trans._(
                    "Dock widget {number}",
                    number=self._unnamed_dockwidget_count,
                )

            self._unnamed_dockwidget_count += 1
        if shortcut is not _sentinel:
            warnings.warn(
                _SHORTCUT_DEPRECATION_STRING.format(shortcut=shortcut),
                FutureWarning,
                stacklevel=2,
            )
            dock_widget = QtViewerDockWidget(
                self.qt_viewer,
                widget,
                name=name,
                area=area,
                allowed_areas=allowed_areas,
                shortcut=shortcut,
                add_vertical_stretch=add_vertical_stretch,
            )
        else:
            dock_widget = QtViewerDockWidget(
                self.qt_viewer,
                widget,
                name=name,
                area=area,
                allowed_areas=allowed_areas,
                add_vertical_stretch=add_vertical_stretch,
            )

        self._add_viewer_dock_widget(dock_widget)

        if hasattr(widget, 'reset_choices'):
            # Keep the dropdown menus in the widget in sync with the layer model
            # if widget has a `reset_choices`, which is true for all magicgui
            # `CategoricalWidget`s
            layers_events = self.qt_viewer.viewer.layers.events
            layers_events.inserted.connect(widget.reset_choices)
            layers_events.removed.connect(widget.reset_choices)
            layers_events.reordered.connect(widget.reset_choices)

        # Add dock widget to dictionary
        self._dock_widgets[dock_widget.name] = dock_widget

        return dock_widget

    def _add_viewer_dock_widget(
        self, dock_widget: QtViewerDockWidget, tabify=False
    ):
        """Add a QtViewerDockWidget to the main window

        If other widgets already present in area then will tabify.

        Parameters
        ----------
        dock_widget : QtViewerDockWidget
            `dock_widget` will be added to the main window.
        tabify : bool
            Flag to tabify dockwidget or not.
        """
        # Find if any othe dock widgets are currently in area
        current_dws_in_area = [
            dw
            for dw in self._qt_window.findChildren(QDockWidget)
            if self._qt_window.dockWidgetArea(dw) == dock_widget.qt_area
        ]
        self._qt_window.addDockWidget(dock_widget.qt_area, dock_widget)

        # If another dock widget present in area then tabify
        if current_dws_in_area:
            if tabify:
                self._qt_window.tabifyDockWidget(
                    current_dws_in_area[-1], dock_widget
                )
                dock_widget.show()
                dock_widget.raise_()
            elif dock_widget.area in ('right', 'left'):
                _wdg = current_dws_in_area + [dock_widget]
                # add sizes to push lower widgets up
                sizes = list(range(1, len(_wdg) * 4, 4))
                self._qt_window.resizeDocks(_wdg, sizes, Qt.Vertical)

        action = dock_widget.toggleViewAction()
        action.setStatusTip(dock_widget.name)
        action.setText(dock_widget.name)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # deprecating with 0.4.8, but let's try to keep compatibility.
            shortcut = dock_widget.shortcut
        if shortcut is not None:
            action.setShortcut(shortcut)
        self.window_menu.addAction(action)

    def remove_dock_widget(self, widget: QWidget):
        """Removes specified dock widget.

        If a QDockWidget is not provided, the existing QDockWidgets will be
        searched for one whose inner widget (``.widget()``) is the provided
        ``widget``.

        Parameters
        ----------
        widget : QWidget | str
            If widget == 'all', all docked widgets will be removed.
        """
        if widget == 'all':
            for dw in list(self._dock_widgets.values()):
                self.remove_dock_widget(dw)
            return

        if not isinstance(widget, QDockWidget):
            for dw in self._qt_window.findChildren(QDockWidget):
                if dw.widget() is widget:
                    _dw: QDockWidget = dw
                    break
            else:
                raise LookupError(
                    trans._(
                        "Could not find a dock widget containing: {widget}",
                        deferred=True,
                        widget=widget,
                    )
                )
        else:
            _dw = widget

        if _dw.widget():
            _dw.widget().setParent(None)
        self._qt_window.removeDockWidget(_dw)
        self.window_menu.removeAction(_dw.toggleViewAction())

        # Remove dock widget from dictionary
        del self._dock_widgets[_dw.name]

        # Deleting the dock widget means any references to it will no longer
        # work but it's not really useful anyway, since the inner widget has
        # been removed. and anyway: people should be using add_dock_widget
        # rather than directly using _add_viewer_dock_widget
        _dw.deleteLater()

    def add_function_widget(
        self,
        function,
        *,
        magic_kwargs=None,
        name: str = '',
        area=None,
        allowed_areas=None,
        shortcut=_sentinel,
    ):
        """Turn a function into a dock widget via magicgui.

        Parameters
        ----------
        function : callable
            Function that you want to add.
        magic_kwargs : dict, optional
            Keyword arguments to :func:`magicgui.magicgui` that
            can be used to specify widget.
        name : str, optional
            Name of dock widget to appear in window menu.
        area : str, optional
            Side of the main window to which the new dock widget will be added.
            Must be in {'left', 'right', 'top', 'bottom'}. If not provided the
            default will be determined by the widget.layout, with 'vertical'
            layouts appearing on the right, otherwise on the bottom.
        allowed_areas : list[str], optional
            Areas, relative to main window, that the widget is allowed dock.
            Each item in list must be in {'left', 'right', 'top', 'bottom'}
            By default, only provided areas is allowed.
        shortcut : str, optional
            Keyboard shortcut to appear in dropdown menu.

        Returns
        -------
        dock_widget : QtViewerDockWidget
            `dock_widget` that can pass viewer events.
        """
        from magicgui import magicgui

        if magic_kwargs is None:
            magic_kwargs = {
                'auto_call': False,
                'call_button': "run",
                'layout': 'vertical',
            }

        widget = magicgui(function, **magic_kwargs or {})

        if area is None:
            if str(widget.layout) == 'vertical':
                area = 'right'
            else:
                area = 'bottom'

        if allowed_areas is None:
            allowed_areas = [area]
        if shortcut is not _sentinel:
            return self.add_dock_widget(
                widget,
                name=name or function.__name__.replace('_', ' '),
                area=area,
                allowed_areas=allowed_areas,
                shortcut=shortcut,
            )
        else:
            return self.add_dock_widget(
                widget,
                name=name or function.__name__.replace('_', ' '),
                area=area,
                allowed_areas=allowed_areas,
            )

    def resize(self, width, height):
        """Resize the window.

        Parameters
        ----------
        width : int
            Width in logical pixels.
        height : int
            Height in logical pixels.
        """
        self._qt_window.resize(width, height)

    def show(self, *, block=False):
        """Resize, show, and bring forward the window.

        Raises
        ------
        RuntimeError
            If the viewer.window has already been closed and deleted.
        """
        settings = get_settings()
        try:
            self._qt_window.show(block=block)
        except (AttributeError, RuntimeError):
            raise RuntimeError(
                trans._(
                    "This viewer has already been closed and deleted. Please create a new one.",
                    deferred=True,
                )
            )

        if settings.application.first_time:
            settings.application.first_time = False
            try:
                self._qt_window.resize(self._qt_window.layout().sizeHint())
            except (AttributeError, RuntimeError):
                raise RuntimeError(
                    trans._(
                        "This viewer has already been closed and deleted. Please create a new one.",
                        deferred=True,
                    )
                )
        else:
            try:
                if settings.application.save_window_geometry:
                    self._qt_window._set_window_settings(
                        *self._qt_window._load_window_settings()
                    )
            except Exception as err:
                import warnings

                warnings.warn(
                    trans._(
                        "The window geometry settings could not be loaded due to the following error: {err}",
                        deferred=True,
                        err=err,
                    ),
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        # Resize axis labels now that window is shown
        self.qt_viewer.dims._resize_axis_labels()

        # We want to bring the viewer to the front when
        # A) it is our own event loop OR we are running in jupyter
        # B) it is not the first time a QMainWindow is being created

        # `app_name` will be "napari" iff the application was instantiated in
        # get_app(). isActiveWindow() will be True if it is the second time a
        # _qt_window has been created.
        # See #721, #732, #735, #795, #1594
        app_name = QApplication.instance().applicationName()
        if (
            app_name == 'napari' or in_jupyter()
        ) and self._qt_window.isActiveWindow():
            self.activate()

    def activate(self):
        """Make the viewer the currently active window."""
        self._qt_window.raise_()  # for macOS
        self._qt_window.activateWindow()  # for Windows

    def _update_theme(self, event=None):
        """Update widget color theme."""
        settings = get_settings()
        if event:
            value = event.value
            settings.appearance.theme = value
            self.qt_viewer.viewer.theme = value
        else:
            value = self.qt_viewer.viewer.theme

        try:
            self._qt_window.setStyleSheet(get_stylesheet(value))
        except AttributeError:
            pass
        except RuntimeError:
            # wrapped C/C++ object may have been deleted
            pass

    def _status_changed(self, event):
        """Update status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._status_bar.showMessage(event.value)

    def _title_changed(self, event):
        """Update window title.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._qt_window.setWindowTitle(event.value)

    def _help_changed(self, event):
        """Update help message on status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._help.setText(event.value)

    def _screenshot_dialog(self):
        """Save screenshot of current display with viewer, default .png"""
        hist = get_save_history()
        dial = ScreenshotDialog(self.screenshot, self.qt_viewer, hist[0], hist)

        if dial.exec_():
            update_save_history(dial.selectedFiles()[0])

    def _restart(self):
        """Restart the napari application."""
        self._qt_window.restart()

    def _screenshot(self, flash=True):
        """Capture screenshot of the currently displayed viewer.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        img = self._qt_window.grab().toImage()
        if flash:
            from .utils import add_flash_animation

            add_flash_animation(self._qt_window)
        return img

    def screenshot(self, path=None, flash=True):
        """Take currently displayed viewer and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        img = self._screenshot(flash)
        if path is not None:
            imsave(path, QImg2array(img))  # scikit-image imsave method
        return QImg2array(img)

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

    def close(self):
        """Close the viewer window and cleanup sub-widgets."""
        # Someone is closing us twice? Only try to delete self._qt_window
        # if we still have one.
        if hasattr(self, '_qt_window'):
            self.qt_viewer.close()
            self._qt_window.close()
            del self._qt_window

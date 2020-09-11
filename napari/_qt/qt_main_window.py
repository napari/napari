"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
import time

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QShortcut,
    QStatusBar,
    QWidget,
)

from ..resources import get_stylesheet
from ..utils import perf
from ..utils.io import imsave
from ..utils.misc import in_jupyter
from ..utils.theme import template
from .dialogs.qt_about import QtAbout
from .dialogs.qt_plugin_report import QtPluginErrReporter
from .dialogs.qt_plugin_table import QtPluginTable
from .qt_viewer import QtViewer
from .tracing.qt_debug_menu import DebugMenu
from .utils import QImg2array
from .widgets.qt_plugin_sorter import QtPluginSorter
from .widgets.qt_viewer_dock_widget import QtViewerDockWidget


class Window:
    """Application window that contains the menu bar and viewer.

    Parameters
    ----------
    qt_viewer : QtViewer
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

    raw_stylesheet = get_stylesheet()

    def __init__(self, qt_viewer: QtViewer, *, show: bool = True):

        self.qt_viewer = qt_viewer

        self._qt_window = QMainWindow()
        self._qt_window.setAttribute(Qt.WA_DeleteOnClose)
        self._qt_window.setUnifiedTitleAndToolBarOnMac(True)
        self._qt_center = QWidget(self._qt_window)

        self._qt_window.setCentralWidget(self._qt_center)
        self._qt_window.setWindowTitle(self.qt_viewer.viewer.title)
        self._qt_center.setLayout(QHBoxLayout())
        self._status_bar = QStatusBar()
        self._qt_window.setStatusBar(self._status_bar)

        self._add_menubar()

        self._add_file_menu()
        self._add_view_menu()
        self._add_window_menu()
        self._add_plugins_menu()
        self._add_help_menu()

        self._status_bar.showMessage('Ready')
        self._help = QLabel('')
        self._status_bar.addPermanentWidget(self._help)

        self._qt_center.layout().addWidget(self.qt_viewer)
        self._qt_center.layout().setContentsMargins(4, 0, 4, 0)

        self._update_palette()

        self._add_viewer_dock_widget(self.qt_viewer.dockConsole)
        self._add_viewer_dock_widget(self.qt_viewer.dockLayerControls)
        self._add_viewer_dock_widget(self.qt_viewer.dockLayerList)

        self.qt_viewer.viewer.events.status.connect(self._status_changed)
        self.qt_viewer.viewer.events.help.connect(self._help_changed)
        self.qt_viewer.viewer.events.title.connect(self._title_changed)
        self.qt_viewer.viewer.events.palette.connect(self._update_palette)

        if perf.USE_PERFMON:
            # Add DebugMenu if using perfmon. The DebugMenu is intended to
            # contain non-perfmon stuff as well. When it does we will want
            # a separate env variable for it.
            self._debug_menu = DebugMenu(self)

            # The QtPerformance widget only exists if we are using perfmon.
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
        open_images = QAction('Open File(s)...', self._qt_window)
        open_images.setShortcut('Ctrl+O')
        open_images.setStatusTip('Open file(s)')
        open_images.triggered.connect(self.qt_viewer._open_files_dialog)

        open_stack = QAction('Open Files as Stack...', self._qt_window)
        open_stack.setShortcut('Ctrl+Alt+O')
        open_stack.setStatusTip('Open files')
        open_stack.triggered.connect(
            self.qt_viewer._open_files_dialog_as_stack_dialog
        )

        open_folder = QAction('Open Folder...', self._qt_window)
        open_folder.setShortcut('Ctrl+Shift+O')
        open_folder.setStatusTip('Open a folder')
        open_folder.triggered.connect(self.qt_viewer._open_folder_dialog)

        save_selected_layers = QAction(
            'Save Selected Layer(s)...', self._qt_window
        )
        save_selected_layers.setShortcut('Ctrl+S')
        save_selected_layers.setStatusTip('Save selected layers')
        save_selected_layers.triggered.connect(
            lambda: self.qt_viewer._save_layers_dialog(selected=True)
        )

        save_all_layers = QAction('Save All Layers...', self._qt_window)
        save_all_layers.setShortcut('Ctrl+Shift+S')
        save_all_layers.setStatusTip('Save all layers')
        save_all_layers.triggered.connect(
            lambda: self.qt_viewer._save_layers_dialog(selected=False)
        )

        screenshot = QAction('Save Screenshot...', self._qt_window)
        screenshot.setShortcut('Alt+S')
        screenshot.setStatusTip(
            'Save screenshot of current display, default .png'
        )
        screenshot.triggered.connect(self.qt_viewer._screenshot_dialog)

        screenshot_wv = QAction(
            'Save Screenshot with Viewer...', self._qt_window
        )
        screenshot_wv.setShortcut('Alt+Shift+S')
        screenshot_wv.setStatusTip(
            'Save screenshot of current display with the viewer, default .png'
        )
        screenshot_wv.triggered.connect(self._screenshot_dialog)

        # OS X will rename this to Quit and put it in the app menu.
        exitAction = QAction('Exit', self._qt_window)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setMenuRole(QAction.QuitRole)

        def handle_exit():
            # if the event loop was started in gui_qt() then the app will be
            # named 'napari'. Since the Qapp was started by us, just close it.
            if QApplication.applicationName() == 'napari':
                QApplication.closeAllWindows()
                QApplication.quit()
            # otherwise, something else created the QApp before us (such as
            # %gui qt IPython magic).  If we quit the app in this case, then
            # *later* attempts to instantiate a napari viewer won't work until
            # the event loop is restarted with app.exec_().  So rather than
            # quit just close all the windows (and clear our app icon).
            else:
                QApplication.setWindowIcon(QIcon())
                self.close()

            if perf.USE_PERFMON:
                # Write trace file before exit, if we were writing one.
                # Is there a better place to make sure this is done on exit?
                perf.timers.stop_trace_file()

        exitAction.triggered.connect(handle_exit)

        self.file_menu = self.main_menu.addMenu('&File')
        self.file_menu.addAction(open_images)
        self.file_menu.addAction(open_stack)
        self.file_menu.addAction(open_folder)
        self.file_menu.addSeparator()
        self.file_menu.addAction(save_selected_layers)
        self.file_menu.addAction(save_all_layers)
        self.file_menu.addAction(screenshot)
        self.file_menu.addAction(screenshot_wv)
        self.file_menu.addSeparator()
        self.file_menu.addAction(exitAction)

    def _add_view_menu(self):
        """Add 'View' menu to app menubar."""
        toggle_visible = QAction('Toggle Menubar Visibility', self._qt_window)
        toggle_visible.setShortcut('Ctrl+M')
        toggle_visible.setStatusTip('Hide Menubar')
        toggle_visible.triggered.connect(self._toggle_menubar_visible)
        toggle_theme = QAction('Toggle Theme', self._qt_window)
        toggle_theme.setShortcut('Ctrl+Shift+T')
        toggle_theme.setStatusTip('Toggle theme')
        toggle_theme.triggered.connect(self.qt_viewer.viewer._toggle_theme)
        self.view_menu = self.main_menu.addMenu('&View')
        self.view_menu.addAction(toggle_visible)
        self.view_menu.addAction(toggle_theme)

    def _add_window_menu(self):
        """Add 'Window' menu to app menubar."""
        exit_action = QAction("Close Window", self._qt_window)
        exit_action.setShortcut("Ctrl+W")
        exit_action.setStatusTip('Close napari window')
        exit_action.triggered.connect(self._qt_window.close)
        self.window_menu = self.main_menu.addMenu('&Window')
        self.window_menu.addAction(exit_action)

    def _add_plugins_menu(self):
        """Add 'Plugins' menu to app menubar."""
        self.plugins_menu = self.main_menu.addMenu('&Plugins')

        list_plugins_action = QAction(
            "List Installed Plugins...", self._qt_window
        )
        list_plugins_action.setStatusTip('List installed plugins')
        list_plugins_action.triggered.connect(self._show_plugin_list)
        self.plugins_menu.addAction(list_plugins_action)

        pip_install_action = QAction(
            "Install/Uninstall Package(s)...", self._qt_window
        )
        pip_install_action.triggered.connect(self._show_pip_install_dialog)
        self.plugins_menu.addAction(pip_install_action)

        order_plugin_action = QAction("Plugin Call Order...", self._qt_window)
        order_plugin_action.setStatusTip('Change call order for plugins')
        order_plugin_action.triggered.connect(self._show_plugin_sorter)
        self.plugins_menu.addAction(order_plugin_action)

        report_plugin_action = QAction("Plugin Errors...", self._qt_window)
        report_plugin_action.setStatusTip(
            'Review stack traces for plugin exceptions and notify developers'
        )
        report_plugin_action.triggered.connect(self._show_plugin_err_reporter)
        self.plugins_menu.addAction(report_plugin_action)

    def _show_plugin_list(self, plugin_manager=None):
        """Show dialog with a table of installed plugins and metadata."""
        QtPluginTable(self._qt_window).exec_()

    def _show_plugin_sorter(self):
        """Show dialog that allows users to sort the call order of plugins."""
        plugin_sorter = QtPluginSorter(parent=self._qt_window)
        dock_widget = self.add_dock_widget(
            plugin_sorter, name='Plugin Sorter', area="right"
        )
        plugin_sorter.finished.connect(dock_widget.close)
        plugin_sorter.finished.connect(plugin_sorter.deleteLater)
        plugin_sorter.finished.connect(dock_widget.deleteLater)

    def _show_pip_install_dialog(self):
        """Show dialog that allows users to sort the call order of plugins."""
        from .qt_pip_dialog import QtPipDialog

        dialog = QtPipDialog(self._qt_window)
        dialog.exec_()

    def _show_plugin_err_reporter(self):
        """Show dialog that allows users to review and report plugin errors."""
        plugin_sorter = QtPluginErrReporter(parent=self._qt_window)
        plugin_sorter.exec_()

    def _add_help_menu(self):
        """Add 'Help' menu to app menubar."""
        self.help_menu = self.main_menu.addMenu('&Help')

        about_action = QAction("napari Info", self._qt_window)
        about_action.setShortcut("Ctrl+/")
        about_action.setStatusTip('About napari')
        about_action.triggered.connect(
            lambda e: QtAbout.showAbout(self.qt_viewer)
        )
        self.help_menu.addAction(about_action)

        about_key_bindings = QAction("Show Key Bindings", self._qt_window)
        about_key_bindings.setShortcut("Ctrl+Alt+/")
        about_key_bindings.setShortcutContext(Qt.ApplicationShortcut)
        about_key_bindings.setStatusTip('key_bindings')
        about_key_bindings.triggered.connect(
            self.qt_viewer.show_key_bindings_dialog
        )
        self.help_menu.addAction(about_key_bindings)

    def add_dock_widget(
        self,
        widget: QWidget,
        *,
        name: str = '',
        area: str = 'bottom',
        allowed_areas=None,
        shortcut=None,
    ):
        """Convenience method to add a QDockWidget to the main window

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

        Returns
        -------
        dock_widget : QtViewerDockWidget
            `dock_widget` that can pass viewer events.
        """

        dock_widget = QtViewerDockWidget(
            self.qt_viewer,
            widget,
            name=name,
            area=area,
            allowed_areas=allowed_areas,
            shortcut=shortcut,
        )
        self._add_viewer_dock_widget(dock_widget)
        return dock_widget

    def _add_viewer_dock_widget(self, dock_widget: QtViewerDockWidget):
        """Add a QtViewerDockWidget to the main window

        Parameters
        ----------
        dock_widget : QtViewerDockWidget
            `dock_widget` will be added to the main window.
        """
        dock_widget.setParent(self._qt_window)
        self._qt_window.addDockWidget(dock_widget.qt_area, dock_widget)
        action = dock_widget.toggleViewAction()
        action.setStatusTip(dock_widget.name)
        action.setText(dock_widget.name)
        if dock_widget.shortcut is not None:
            action.setShortcut(dock_widget.shortcut)
        self.window_menu.addAction(action)

    def remove_dock_widget(self, widget):
        """Removes specified dock widget.

        Parameters
        ----------
        widget : QWidget | str
            If widget == 'all', all docked widgets will be removed.
        """
        if widget == 'all':
            for dw in self._qt_window.findChildren(QDockWidget):
                self._qt_window.removeDockWidget(dw)
        else:
            self._qt_window.removeDockWidget(widget)

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

    def show(self):
        """Resize, show, and bring forward the window."""
        self._qt_window.resize(self._qt_window.layout().sizeHint())
        self._qt_window.show()
        # Resize axis labels now that window is shown
        self.qt_viewer.dims._resize_axis_labels()

        # We want to bring the viewer to the front when
        # A) it is our own (gui_qt) event loop OR we are running in jupyter
        # B) it is not the first time a QMainWindow is being created

        # `app_name` will be "napari" iff the application was instantiated in
        # gui_qt(). isActiveWindow() will be True if it is the second time a
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

    def _update_palette(self, event=None):
        """Update widget color palette."""
        # set window styles which don't use the primary stylesheet
        # FIXME: this is a problem with the stylesheet not using properties
        palette = self.qt_viewer.viewer.palette
        self._status_bar.setStyleSheet(
            template(
                'QStatusBar { background: {{ background }}; '
                'color: {{ text }}; }',
                **palette,
            )
        )
        self._qt_center.setStyleSheet(
            template('QWidget { background: {{ background }}; }', **palette)
        )
        self._qt_window.setStyleSheet(template(self.raw_stylesheet, **palette))

    def _status_changed(self, event):
        """Update status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._status_bar.showMessage(event.text)

    def _title_changed(self, event):
        """Update window title.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._qt_window.setWindowTitle(event.text)

    def _help_changed(self, event):
        """Update help message on status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._help.setText(event.text)

    def _screenshot_dialog(self):
        """Save screenshot of current display with viewer, default .png"""
        filename, _ = QFileDialog.getSaveFileName(
            parent=self.qt_viewer,
            caption='Save screenshot with viewer',
            directory=self.qt_viewer._last_visited_dir,  # home dir by default
            filter="Image files (*.png *.bmp *.gif *.tif *.tiff)",  # first one used by default
            # jpg and jpeg not included as they don't support an alpha channel
        )
        if (filename != '') and (filename is not None):
            # double check that an appropriate extension has been added as the
            # filter option does not always add an extension on linux and windows
            # see https://bugreports.qt.io/browse/QTBUG-27186
            image_extensions = ('.bmp', '.gif', '.png', '.tif', '.tiff')
            if not filename.endswith(image_extensions):
                filename = filename + '.png'
            self.screenshot(path=filename)

    def screenshot(self, path=None):
        """Take currently displayed viewer and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        img = self._qt_window.grab().toImage()
        if path is not None:
            imsave(path, QImg2array(img))  # scikit-image imsave method
        return QImg2array(img)

    def close(self):
        """Close the viewer window and cleanup sub-widgets."""

        # Someone is closing us twice? Only try to delete self._qt_window
        # if we still have one.
        if hasattr(self, '_qt_window'):
            self._delete_qt_window()

    def _delete_qt_window(self):
        """Delete our self._qt_window."""

        # On some versions of Darwin, exiting while fullscreen seems to tickle
        # some bug deep in NSWindow.  This forces the fullscreen keybinding
        # test to complete its draw cycle, then pop back out of fullscreen.
        if self._qt_window.isFullScreen():
            self._qt_window.showNormal()
            for i in range(8):
                time.sleep(0.1)
                QApplication.processEvents()
        self.qt_viewer.close()
        self._qt_window.close()
        del self._qt_window

"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
# set vispy to use same backend as qtpy
import os

from skimage.io import imsave

from .qt_about import QtAbout
from .qt_viewer_dock_widget import QtViewerDockWidget
from ..resources import resources_dir

# these "# noqa" comments are here to skip flake8 linting (E402),
# these module-level imports have to come after `app.use_app(API)`
# see discussion on #638
from qtpy.QtWidgets import (  # noqa: E402
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QDockWidget,
    QLabel,
    QAction,
    QShortcut,
    QStatusBar,
    QApplication,
)
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtGui import QKeySequence  # noqa: E402
from .utils import QImg2array  # noqa: E402
from ..utils.theme import template  # noqa: E402


class Window:
    """Application window that contains the menu bar and viewer.

    Parameters
    ----------
    qt_viewer : QtViewer
        Contained viewer widget.

    Attributes
    ----------
    qt_viewer : QtViewer
        Contained viewer widget.
    """

    with open(os.path.join(resources_dir, 'stylesheet.qss'), 'r') as f:
        raw_stylesheet = f.read()

    def __init__(self, qt_viewer, *, show=True):

        self.qt_viewer = qt_viewer

        self._qt_window = QMainWindow()
        self._qt_window.setUnifiedTitleAndToolBarOnMac(True)
        self._qt_center = QWidget(self._qt_window)

        self._qt_window.setCentralWidget(self._qt_center)
        self._qt_window.setWindowTitle(self.qt_viewer.viewer.title)
        self._qt_center.setLayout(QHBoxLayout())
        self._status_bar = QStatusBar()
        self._qt_window.setStatusBar(self._status_bar)
        self._qt_window.closeEvent = self.closeEvent
        self.close = self._qt_window.close

        self._add_menubar()

        self._add_file_menu()
        self._add_view_menu()
        self._add_window_menu()
        self._add_help_menu()

        self._status_bar.showMessage('Ready')
        self._help = QLabel('')
        self._status_bar.addPermanentWidget(self._help)

        self._qt_center.layout().addWidget(self.qt_viewer)
        self._qt_center.layout().setContentsMargins(4, 0, 4, 0)

        self._update_palette(qt_viewer.viewer.palette)

        if self.qt_viewer.console.shell is not None:
            self._add_viewer_dock_widget(self.qt_viewer.dockConsole)

        self._add_viewer_dock_widget(self.qt_viewer.dockLayerControls)
        self._add_viewer_dock_widget(self.qt_viewer.dockLayerList)

        self.qt_viewer.viewer.events.status.connect(self._status_changed)
        self.qt_viewer.viewer.events.help.connect(self._help_changed)
        self.qt_viewer.viewer.events.title.connect(self._title_changed)
        self.qt_viewer.viewer.events.palette.connect(
            lambda event: self._update_palette(event.palette)
        )

        if show:
            self.show()

    def _add_menubar(self):
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
        open_images = QAction('Open image(s)...', self._qt_window)
        open_images.setShortcut('Ctrl+O')
        open_images.setStatusTip('Open image file(s)')
        open_images.triggered.connect(self.qt_viewer._open_images)

        open_folder = QAction('Open Folder...', self._qt_window)
        open_folder.setShortcut('Ctrl+Shift+O')
        open_folder.setStatusTip(
            'Open a folder of image file(s) or a zarr file'
        )
        open_folder.triggered.connect(self.qt_viewer._open_folder)

        screenshot = QAction('Screenshot', self._qt_window)
        screenshot.setShortcut('Ctrl+Alt+S')
        screenshot.setStatusTip(
            'Save screenshot of current display, default .png'
        )
        screenshot.triggered.connect(self.qt_viewer._save_screenshot)

        self.file_menu = self.main_menu.addMenu('&File')
        self.file_menu.addAction(open_images)
        self.file_menu.addAction(open_folder)
        self.file_menu.addAction(screenshot)

    def _add_view_menu(self):
        toggle_visible = QAction('Toggle menubar visibility', self._qt_window)
        toggle_visible.setShortcut('Ctrl+M')
        toggle_visible.setStatusTip('Hide Menubar')
        toggle_visible.triggered.connect(self._toggle_menubar_visible)
        toggle_theme = QAction('Toggle theme', self._qt_window)
        toggle_theme.setShortcut('Ctrl+Shift+T')
        toggle_theme.setStatusTip('Toggle theme')
        toggle_theme.triggered.connect(self.qt_viewer.viewer._toggle_theme)
        self.view_menu = self.main_menu.addMenu('&View')
        self.view_menu.addAction(toggle_visible)
        self.view_menu.addAction(toggle_theme)

    def _add_window_menu(self):
        exit_action = QAction("Close window", self._qt_window)
        exit_action.setShortcut("Ctrl+W")
        exit_action.setStatusTip('Close napari window')
        exit_action.triggered.connect(self._qt_window.close)
        self.window_menu = self.main_menu.addMenu('&Window')
        self.window_menu.addAction(exit_action)

    def _add_help_menu(self):
        self.help_menu = self.main_menu.addMenu('&Help')

        about_action = QAction("napari info", self._qt_window)
        about_action.setShortcut("Ctrl+/")
        about_action.setStatusTip('About napari')
        about_action.triggered.connect(
            lambda e: QtAbout.showAbout(self.qt_viewer)
        )
        self.help_menu.addAction(about_action)

        about_keybindings = QAction("keybindings", self._qt_window)
        about_keybindings.setShortcut("Ctrl+Alt+/")
        about_keybindings.setShortcutContext(Qt.ApplicationShortcut)
        about_keybindings.setStatusTip('keybindings')
        about_keybindings.triggered.connect(
            self.qt_viewer.aboutKeybindings.toggle_visible
        )
        self.help_menu.addAction(about_keybindings)

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

        # We want to call Window._qt_window.raise_() in every case *except*
        # when instantiating a viewer within a gui_qt() context for the
        # _first_ time within the Qt app's lifecycle.
        #
        # `app_name` will be "napari" iff the application was instantiated in
        # gui_qt(). isActiveWindow() will be True if it is the second time a
        # _qt_window has been created. See #732
        app_name = QApplication.instance().applicationName()
        if app_name != 'napari' or self._qt_window.isActiveWindow():
            self._qt_window.raise_()  # for macOS
            self._qt_window.activateWindow()  # for Windows

    def _update_palette(self, palette):
        # set window styles which don't use the primary stylesheet
        # FIXME: this is a problem with the stylesheet not using properties
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
        """
        self._status_bar.showMessage(event.text)

    def _title_changed(self, event):
        """Update window title.
        """
        self._qt_window.setWindowTitle(event.text)

    def _help_changed(self, event):
        """Update help message on status bar.
        """
        self._help.setText(event.text)

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

    def closeEvent(self, event):
        # Forward close event to the console to trigger proper shutdown
        self.qt_viewer.console.shutdown()
        # if the viewer.QtDims object is playing an axis, we need to terminate the
        # AnimationThread before close, otherwise it will cauyse a segFault or Abort trap.
        # (calling stop() when no animation is occuring is also not a problem)
        self.qt_viewer.dims.stop()
        event.accept()

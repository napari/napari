"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""

import contextlib
import inspect
import os
import sys
import time
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from weakref import WeakValueDictionary

from qtpy.QtCore import QEvent, QEventLoop, QPoint, QProcess, QSize, Qt, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QShortcut,
    QToolTip,
    QWidget,
)
from superqt.utils import QSignalThrottler

from ..plugins import menu_item_template as plugin_menu_item_template
from ..plugins import plugin_manager
from ..settings import get_settings
from ..utils import perf
from ..utils._proxies import PublicOnlyProxy
from ..utils.io import imsave
from ..utils.misc import in_ipython, in_jupyter, running_as_bundled_app
from ..utils.notifications import Notification
from ..utils.theme import _themes, get_system_theme
from ..utils.translations import trans
from . import menus
from .dialogs.confirm_close_dialog import ConfirmCloseDialog
from .dialogs.qt_activity_dialog import QtActivityDialog
from .dialogs.qt_notification import NapariQtNotification
from .qt_event_loop import NAPARI_ICON_PATH, get_app, quit_app
from .qt_resources import get_stylesheet
from .qt_viewer import QtViewer
from .utils import QImg2array, qbytearray_to_str, str_to_qbytearray
from .widgets.qt_viewer_dock_widget import (
    _SHORTCUT_DEPRECATION_STRING,
    QtViewerDockWidget,
)
from .widgets.qt_viewer_status_bar import ViewerStatusBar

_sentinel = object()

if TYPE_CHECKING:
    from magicgui.widgets import Widget
    from qtpy.QtGui import QImage

    from ..viewer import Viewer


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

    def __init__(self, viewer: 'Viewer', parent=None) -> None:
        super().__init__(parent)
        self._ev = None
        self._qt_viewer = QtViewer(viewer, show_welcome_screen=True)
        self._quit_app = False

        self.setWindowIcon(QIcon(self._window_icon))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setUnifiedTitleAndToolBarOnMac(True)
        center = QWidget(self)
        center.setLayout(QHBoxLayout())
        center.layout().addWidget(self._qt_viewer)
        center.layout().setContentsMargins(4, 0, 4, 0)
        self.setCentralWidget(center)

        self.setWindowTitle(self._qt_viewer.viewer.title)

        self._maximized_flag = False
        self._window_size = None
        self._window_pos = None
        self._old_size = None
        self._positions = []

        act_dlg = QtActivityDialog(self._qt_viewer._canvas_overlay)
        self._qt_viewer._canvas_overlay.resized.connect(
            act_dlg.move_to_bottom_right
        )
        act_dlg.hide()
        self._activity_dialog = act_dlg

        self.setStatusBar(ViewerStatusBar(self))

        settings = get_settings()

        # TODO:
        # settings.plugins.defaults.call_order = plugin_manager.call_order()

        # set the values in plugins to match the ones saved in settings
        if settings.plugins.call_order is not None:
            plugin_manager.set_call_order(settings.plugins.call_order)

        _QtMainWindow._instances.append(self)

        # since we initialize canvas before window,
        # we need to manually connect them again.
        handle = self.windowHandle()
        if handle is not None:
            handle.screenChanged.connect(
                self._qt_viewer.canvas._backend.screen_changed
            )

        self.status_throttler = QSignalThrottler(parent=self)
        self.status_throttler.setTimeout(50)

        # Here we disconnect function that update statusbar,
        # and connect it throttled version to get smother GUI experience
        with contextlib.suppress(IndexError):
            viewer.cursor.events.position.disconnect(
                viewer._update_status_bar_from_cursor
            )
        viewer.cursor.events.position.connect(self.status_throttler.throttle)
        self.status_throttler.triggered.connect(
            viewer._update_status_bar_from_cursor
        )

    def statusBar(self) -> 'ViewerStatusBar':
        return super().statusBar()

    @classmethod
    def current(cls) -> Optional['_QtMainWindow']:
        return cls._instances[-1] if cls._instances else None

    @classmethod
    def current_viewer(cls):
        window = cls.current()
        return window._qt_viewer.viewer if window else None

    def event(self, e: QEvent) -> bool:
        if (
            e.type() == QEvent.Type.ToolTip
            and self._qt_viewer.viewer.tooltip.visible
        ):
            # globalPos is for Qt5 e.globalPosition().toPoint() is for QT6
            # https://doc-snapshots.qt.io/qt6-dev/qmouseevent-obsolete.html#globalPos
            pnt = (
                e.globalPosition().toPoint()
                if hasattr(e, "globalPosition")
                else e.globalPos()
            )
            QToolTip.showText(pnt, self._qt_viewer.viewer.tooltip.text, self)
        if e.type() == QEvent.Type.Close:
            # when we close the MainWindow, remove it from the instances list
            with contextlib.suppress(ValueError):
                _QtMainWindow._instances.remove(self)
        if e.type() in {QEvent.Type.WindowActivate, QEvent.Type.ZOrderChange}:
            # upon activation or raise_, put window at the end of _instances
            with contextlib.suppress(ValueError):
                inst = _QtMainWindow._instances
                inst.append(inst.pop(inst.index(self)))
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
            screen_geo = QApplication.primaryScreen().geometry()
            if screen_geo.width() < width or screen_geo.height() < height:
                window_position = (self.x(), self.y())

        return (
            settings.application.window_state,
            settings.application.window_size,
            window_position,
            settings.application.window_maximized,
            settings.application.window_fullscreen,
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

        window_state = qbytearray_to_str(self.saveState())
        return (
            window_state,
            self._window_size or (self.width(), self.height()),
            self._window_pos or (self.x(), self.y()),
            window_maximized,
            window_fullscreen,
        )

    def _set_window_settings(
        self,
        window_state,
        window_size,
        window_position,
        window_maximized,
        window_fullscreen,
    ):
        """
        Set window settings.

        Symmetric to the 'get_window_settings' accessor.
        """
        self.setUpdatesEnabled(False)
        self.setWindowState(Qt.WindowState.WindowNoState)

        if window_position:
            window_position = QPoint(*window_position)
            self.move(window_position)

        if window_size:
            window_size = QSize(*window_size)
            self.resize(window_size)

        if window_state:
            self.restoreState(str_to_qbytearray(window_state))

        # Toggling the console visibility is disabled when it is not
        # available, so ensure that it is hidden.
        if in_ipython():
            self._qt_viewer.dockConsole.setVisible(False)

        if window_fullscreen:
            self.setWindowState(Qt.WindowState.WindowFullScreen)
            self._maximized_flag = window_maximized
        elif window_maximized:
            self.setWindowState(Qt.WindowState.WindowMaximized)

        self.setUpdatesEnabled(True)

    def _save_current_window_settings(self):
        """Save the current geometry of the main window."""
        (
            window_state,
            window_size,
            window_position,
            window_maximized,
            window_fullscreen,
        ) = self._get_window_settings()

        settings = get_settings()
        if settings.application.save_window_geometry:
            settings.application.window_maximized = window_maximized
            settings.application.window_fullscreen = window_fullscreen
            settings.application.window_position = window_position
            settings.application.window_size = window_size
            settings.application.window_statusbar = (
                not self.statusBar().isHidden()
            )

        if settings.application.save_window_state:
            settings.application.window_state = window_state

    def close(self, quit_app=False, confirm_need=False):
        """Override to handle closing app or just the window."""
        if hasattr(self.status_throttler, "_timer"):
            self.status_throttler._timer.stop()
        if not quit_app and not self._qt_viewer.viewer.layers:
            return super().close()
        if (
            not confirm_need
            or not get_settings().application.confirm_close_window
            or ConfirmCloseDialog(self, quit_app).exec_() == QDialog.Accepted
        ):
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
            except AttributeError:
                parent = getattr(parent, "_parent", None)

    def show(self, block=False):
        super().show()
        self._qt_viewer.setFocus()
        if block:
            self._ev = QEventLoop()
            self._ev.exec()

    def changeEvent(self, event):
        """Handle window state changes."""
        if event.type() == QEvent.Type.WindowStateChange:
            # TODO: handle maximization issue. When double clicking on the
            # title bar on Mac the resizeEvent is called an varying amount
            # of times which makes it hard to track the original size before
            # maximization.
            condition = (
                self.isMaximized() if os.name == "nt" else self.isFullScreen()
            )
            if condition and self._old_size is not None:
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
        if (
            event.spontaneous()
            and get_settings().application.confirm_close_window
            and self._qt_viewer.viewer.layers
            and ConfirmCloseDialog(self, False).exec_() != QDialog.Accepted
        ):
            event.ignore()
            return

        if self._ev and self._ev.isRunning():
            self._ev.quit()

        # Close any floating dockwidgets
        for dock in self.findChildren(QtViewerDockWidget):
            if isinstance(dock, QWidget) and dock.isFloating():
                dock.setFloating(False)

        self._save_current_window_settings()

        # On some versions of Darwin, exiting while fullscreen seems to tickle
        # some bug deep in NSWindow.  This forces the fullscreen keybinding
        # test to complete its draw cycle, then pop back out of fullscreen.
        if self.isFullScreen():
            self.showNormal()
            for _ in range(5):
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

    view_menu : qtpy.QtWidgets.QMenu
        View menu.
    window_menu : qtpy.QtWidgets.QMenu
        Window menu.
    """

    def __init__(self, viewer: 'Viewer', *, show: bool = True):
        # create QApplication if it doesn't already exist
        get_app()

        # Dictionary holding dock widgets
        self._dock_widgets: Dict[
            str, QtViewerDockWidget
        ] = WeakValueDictionary()
        self._unnamed_dockwidget_count = 1

        # Connect the Viewer and create the Main Window
        self._qt_window = _QtMainWindow(viewer)

        # connect theme events before collecting plugin-provided themes
        # to ensure icons from the plugins are generated correctly.
        _themes.events.added.connect(self._add_theme)
        _themes.events.removed.connect(self._remove_theme)

        # discover any themes provided by plugins
        plugin_manager.discover_themes()
        self._setup_existing_themes()

        self._add_menus()
        self._update_theme()
        get_settings().appearance.events.theme.connect(self._update_theme)

        self._add_viewer_dock_widget(
            self._qt_viewer.dockConsole, tabify=False, menu=self.window_menu
        )
        self._add_viewer_dock_widget(
            self._qt_viewer.dockLayerControls,
            tabify=False,
            menu=self.window_menu,
        )
        self._add_viewer_dock_widget(
            self._qt_viewer.dockLayerList, tabify=False, menu=self.window_menu
        )
        if perf.USE_PERFMON:
            self._add_viewer_dock_widget(
                self._qt_viewer.dockPerformance, menu=self.window_menu
            )

        viewer.events.help.connect(self._help_changed)
        viewer.events.title.connect(self._title_changed)
        viewer.events.theme.connect(self._update_theme)
        viewer.layers.events.connect(self.file_menu.update)
        viewer.events.status.connect(self._status_changed)

        if show:
            self.show()
            # Ensure the controls dock uses the minimum height
            self._qt_window.resizeDocks(
                [
                    self._qt_viewer.dockLayerControls,
                    self._qt_viewer.dockLayerList,
                ],
                [self._qt_viewer.dockLayerControls.minimumHeight(), 10000],
                Qt.Orientation.Vertical,
            )

    def _setup_existing_themes(self, connect: bool = True):
        """This function is only executed once at the startup of napari
        to connect events to themes that have not been connected yet.

        Parameters
        ----------
        connect : bool
            Determines whether the `connect` or `disconnect` method should be used.
        """
        for theme in _themes.values():
            if connect:
                self._connect_theme(theme)
            else:
                self._disconnect_theme(theme)

    def _connect_theme(self, theme):
        # connect events to update theme. Here, we don't want to pass the event
        # since it won't have the right `value` attribute.
        theme.events.background.connect(self._update_theme_no_event)
        theme.events.foreground.connect(self._update_theme_no_event)
        theme.events.primary.connect(self._update_theme_no_event)
        theme.events.secondary.connect(self._update_theme_no_event)
        theme.events.highlight.connect(self._update_theme_no_event)
        theme.events.text.connect(self._update_theme_no_event)
        theme.events.warning.connect(self._update_theme_no_event)
        theme.events.current.connect(self._update_theme_no_event)
        theme.events.icon.connect(self._update_theme_no_event)
        theme.events.canvas.connect(
            lambda _: self._qt_viewer.canvas._set_theme_change(
                get_settings().appearance.theme
            )
        )
        # connect console-specific attributes only if QtConsole
        # is present. The `console` is called which might slow
        # things down a little.
        if self._qt_viewer._console:
            theme.events.console.connect(self._qt_viewer.console._update_theme)
            theme.events.syntax_style.connect(
                self._qt_viewer.console._update_theme
            )

    def _disconnect_theme(self, theme):
        theme.events.background.disconnect(self._update_theme_no_event)
        theme.events.foreground.disconnect(self._update_theme_no_event)
        theme.events.primary.disconnect(self._update_theme_no_event)
        theme.events.secondary.disconnect(self._update_theme_no_event)
        theme.events.highlight.disconnect(self._update_theme_no_event)
        theme.events.text.disconnect(self._update_theme_no_event)
        theme.events.warning.disconnect(self._update_theme_no_event)
        theme.events.current.disconnect(self._update_theme_no_event)
        theme.events.icon.disconnect(self._update_theme_no_event)
        theme.events.canvas.disconnect(
            lambda _: self._qt_viewer.canvas._set_theme_change(
                get_settings().appearance.theme
            )
        )
        # disconnect console-specific attributes only if QtConsole
        # is present and they were previously connected
        if self._qt_viewer._console:
            theme.events.console.disconnect(
                self._qt_viewer.console._update_theme
            )
            theme.events.syntax_style.disconnect(
                self._qt_viewer.console._update_theme
            )

    def _add_theme(self, event):
        """Add new theme and connect events."""
        theme = event.value
        self._connect_theme(theme)

    def _remove_theme(self, event):
        """Remove theme and disconnect events."""
        theme = event.value
        self._disconnect_theme(theme)

    @property
    def qt_viewer(self):
        warnings.warn(
            trans._(
                'Public access to Window.qt_viewer is deprecated and will be removed in\nv0.5.0. It is considered an "implementation detail" of the napari\napplication, not part of the napari viewer model. If your use case\nrequires access to qt_viewer, please open an issue to discuss.',
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self._qt_window._qt_viewer

    @property
    def _qt_viewer(self):
        # this is starting to be "vestigial"... this property could be removed
        return self._qt_window._qt_viewer

    @property
    def _status_bar(self):
        # TODO: remove from window
        return self._qt_window.statusBar()

    def _add_menus(self):
        """Add menubar to napari app."""
        # TODO: move this to _QMainWindow... but then all of the Menu()
        # items will not have easy access to the methods on this Window obj.

        self.main_menu = self._qt_window.menuBar()
        # Menubar shortcuts are only active when the menubar is visible.
        # Therefore, we set a global shortcut not associated with the menubar
        # to toggle visibility, *but*, in order to not shadow the menubar
        # shortcut, we disable it, and only enable it when the menubar is
        # hidden. See this stackoverflow link for details:
        # https://stackoverflow.com/questions/50537642/how-to-keep-the-shortcuts-of-a-hidden-widget-in-pyqt5
        self._main_menu_shortcut = QShortcut('Ctrl+M', self._qt_window)
        self._main_menu_shortcut.setEnabled(False)
        self._main_menu_shortcut.activated.connect(
            self._toggle_menubar_visible
        )

        self.file_menu = menus.FileMenu(self)
        self.main_menu.addMenu(self.file_menu)
        self.view_menu = menus.ViewMenu(self)
        self.main_menu.addMenu(self.view_menu)
        self.window_menu = menus.WindowMenu(self)
        self.main_menu.addMenu(self.window_menu)
        self.plugins_menu = menus.PluginsMenu(self)
        self.main_menu.addMenu(self.plugins_menu)
        self.help_menu = menus.HelpMenu(self)
        self.main_menu.addMenu(self.help_menu)

        if perf.USE_PERFMON:
            self._debug_menu = menus.DebugMenu(self)
            self.main_menu.addMenu(self._debug_menu)

    def _toggle_menubar_visible(self):
        """Toggle visibility of app menubar.

        This function also disables or enables a global keyboard shortcut to
        show the menubar, since menubar shortcuts are only available while the
        menubar is visible.
        """
        self.main_menu.setVisible(not self.main_menu.isVisible())
        self._main_menu_shortcut.setEnabled(not self.main_menu.isVisible())

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self._qt_window.isFullScreen():
            self._qt_window.showNormal()
        else:
            self._qt_window.showFullScreen()

    def _toggle_play(self):
        """Toggle play."""
        if self._qt_viewer.dims.is_playing:
            self._qt_viewer.dims.stop()
        else:
            axis = self._qt_viewer.viewer.dims.last_used or 0
            self._qt_viewer.dims.play(axis)

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
        from ..plugins import _npe2

        Widget = None
        dock_kwargs = {}

        if result := _npe2.get_widget_contribution(plugin_name, widget_name):
            Widget, widget_name = result

        if Widget is None:
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
            wdg = dock_widget.widget()
            if hasattr(wdg, '_magic_widget'):
                wdg = wdg._magic_widget
            return dock_widget, wdg

        wdg = _instantiate_dock_widget(
            Widget, cast('Viewer', self._qt_viewer.viewer)
        )

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
            return

        func = plugin_manager._function_widgets[plugin_name][widget_name]

        # Add function widget
        return self.add_function_widget(
            func, name=full_name, area=None, allowed_areas=None
        )

    def add_dock_widget(
        self,
        widget: Union[QWidget, 'Widget'],
        *,
        name: str = '',
        area: str = 'right',
        allowed_areas: Optional[Sequence[str]] = None,
        shortcut=_sentinel,
        add_vertical_stretch=True,
        menu=None,
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
            with contextlib.suppress(AttributeError):
                name = widget.objectName()
            name = name or trans._(
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
                self._qt_viewer,
                widget,
                name=name,
                area=area,
                allowed_areas=allowed_areas,
                shortcut=shortcut,
                add_vertical_stretch=add_vertical_stretch,
            )
        else:
            dock_widget = QtViewerDockWidget(
                self._qt_viewer,
                widget,
                name=name,
                area=area,
                allowed_areas=allowed_areas,
                add_vertical_stretch=add_vertical_stretch,
            )

        self._add_viewer_dock_widget(dock_widget, menu=menu)

        if hasattr(widget, 'reset_choices'):
            # Keep the dropdown menus in the widget in sync with the layer model
            # if widget has a `reset_choices`, which is true for all magicgui
            # `CategoricalWidget`s
            layers_events = self._qt_viewer.viewer.layers.events
            layers_events.inserted.connect(widget.reset_choices)
            layers_events.removed.connect(widget.reset_choices)
            layers_events.reordered.connect(widget.reset_choices)

        # Add dock widget to dictionary
        self._dock_widgets[dock_widget.name] = dock_widget

        return dock_widget

    def _add_viewer_dock_widget(
        self, dock_widget: QtViewerDockWidget, tabify=False, menu=None
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
                self._qt_window.resizeDocks(
                    _wdg, sizes, Qt.Orientation.Vertical
                )

        if menu:
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

            menu.addAction(action)
        # self.window_menu.addAction(action)

        # see #3663, to fix #3624 more generally
        dock_widget.setFloating(False)

    def _remove_dock_widget(self, event=None):
        names = list(self._dock_widgets.keys())
        for widget_name in names:
            if event.value in widget_name:
                # remove this widget
                widget = self._dock_widgets[widget_name]
                self.remove_dock_widget(widget)

    def remove_dock_widget(self, widget: QWidget, menu=None):
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
            dw: QDockWidget
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
        if menu is not None:
            menu.removeAction(_dw.toggleViewAction())

        # Remove dock widget from dictionary
        self._dock_widgets.pop(_dw.name, None)

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
            area = 'right' if str(widget.layout) == 'vertical' else 'bottom'
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

    def set_geometry(self, left, top, width, height):
        """Set the geometry of the widget

        Parameters
        ----------
        left : int
            X coordinate of the upper left border.
        top : int
            Y coordinate of the upper left border.
        width : int
            Width of the rectangle shape of the window.
        height : int
            Height of the rectangle shape of the window.
        """
        self._qt_window.setGeometry(left, top, width, height)

    def geometry(self) -> Tuple[int, int, int, int]:
        """Get the geometry of the widget

        Returns
        -------
        left : int
            X coordinate of the upper left border.
        top : int
            Y coordinate of the upper left border.
        width : int
            Width of the rectangle shape of the window.
        height : int
            Height of the rectangle shape of the window.
        """
        rect = self._qt_window.geometry()
        return rect.left(), rect.top(), rect.width(), rect.height()

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
        self._qt_viewer.dims._resize_axis_labels()

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

    def _update_theme_no_event(self):
        self._update_theme()

    def _update_theme(self, event=None):
        """Update widget color theme."""
        settings = get_settings()
        with contextlib.suppress(AttributeError, RuntimeError):
            if event:
                value = event.value
                self._qt_viewer.viewer.theme = value
                settings.appearance.theme = value
            else:
                value = (
                    get_system_theme()
                    if settings.appearance.theme == "system"
                    else self._qt_viewer.viewer.theme
                )

            self._qt_window.setStyleSheet(get_stylesheet(value))

    def _status_changed(self, event):
        """Update status bar.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if isinstance(event.value, str):
            self._status_bar.setStatusText(event.value)
        else:
            status_info = event.value
            self._status_bar.setStatusText(
                layer_base=status_info['layer_base'],
                source_type=status_info['source_type'],
                plugin=status_info['plugin'],
                coordinates=status_info['coordinates'],
            )

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
        self._status_bar.setHelpText(event.value)

    def _restart(self):
        """Restart the napari application."""
        self._qt_window.restart()

    def _screenshot(
        self, size=None, scale=None, flash=True, canvas_only=False
    ) -> 'QImage':
        """Capture screenshot of the currently displayed viewer.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        size : tuple (int, int)
            Size (resolution) of the screenshot. By default, the currently displayed size.
            Only used if `canvas_only` is True.
        scale : float
            Scale factor used to increase resolution of canvas for the screenshot. By default, the currently displayed resolution.
            Only used if `canvas_only` is True.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.

        Returns
        -------
        img : QImage
        """
        from .utils import add_flash_animation

        if canvas_only:
            canvas = self._qt_viewer.canvas
            prev_size = canvas.size
            if size is not None:
                if len(size) != 2:
                    raise ValueError(
                        f'screenshot size must be 2 values, got {len(size)}'
                    )
                # Scale the requested size to account for HiDPI
                size = tuple(
                    dim / self._qt_window.devicePixelRatio() for dim in size
                )
                canvas.size = size[::-1]  # invert x ad y for vispy
            if scale is not None:
                # multiply canvas dimensions by the scale factor to get new size
                canvas.size = tuple(dim * scale for dim in canvas.size)
            try:
                img = self._qt_viewer.canvas.native.grabFramebuffer()
                if flash:
                    add_flash_animation(self._qt_viewer._canvas_overlay)
            finally:
                # make sure we always go back to the right canvas size
                if size is not None or scale is not None:
                    canvas.size = prev_size
        else:
            img = self._qt_window.grab().toImage()
            if flash:
                add_flash_animation(self._qt_window)
        return img

    def screenshot(
        self, path=None, size=None, scale=None, flash=True, canvas_only=False
    ):
        """Take currently displayed viewer and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        size : tuple (int, int)
            Size (resolution) of the screenshot. By default, the currently displayed size.
            Only used if `canvas_only` is True.
        scale : float
            Scale factor used to increase resolution of canvas for the screenshot. By default, the currently displayed resolution.
            Only used if `canvas_only` is True.
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        img = QImg2array(self._screenshot(size, scale, flash, canvas_only))
        if path is not None:
            imsave(path, img)  # scikit-image imsave method
        return img

    def clipboard(self, flash=True, canvas_only=False):
        """Copy screenshot of current viewer to the clipboard.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.
        """
        img = self._screenshot(flash=flash, canvas_only=canvas_only)
        QApplication.clipboard().setImage(img)

    def _teardown(self):
        """Carry out various teardown tasks such as event disconnection."""
        self._setup_existing_themes(False)
        _themes.events.added.disconnect(self._add_theme)
        _themes.events.removed.disconnect(self._remove_theme)
        self._qt_viewer.viewer.layers.events.disconnect(self.file_menu.update)
        for menu in self.file_menu._INSTANCES:
            with contextlib.suppress(RuntimeError):
                menu._destroy()

    def close(self):
        """Close the viewer window and cleanup sub-widgets."""
        # Someone is closing us twice? Only try to delete self._qt_window
        # if we still have one.
        if hasattr(self, '_qt_window'):
            # disconnect events to prevent leaking `Viewer` object because of throttle
            self._teardown()
            self._qt_viewer.close()
            self._qt_window.close()
            del self._qt_window


def _instantiate_dock_widget(wdg_cls, viewer: 'Viewer'):
    # if the signature is looking a for a napari viewer, pass it.
    from ..viewer import Viewer

    kwargs = {}
    try:
        sig = inspect.signature(wdg_cls.__init__)
    except ValueError:
        pass
    else:
        for param in sig.parameters.values():
            if param.name == 'napari_viewer':
                kwargs['napari_viewer'] = PublicOnlyProxy(viewer)
                break
            if param.annotation in ('napari.viewer.Viewer', Viewer):
                kwargs[param.name] = PublicOnlyProxy(viewer)
                break
            # cannot look for param.kind == param.VAR_KEYWORD because
            # QWidget allows **kwargs but errs on unknown keyword arguments

    # instantiate the widget
    return wdg_cls(**kwargs)

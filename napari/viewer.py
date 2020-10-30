import platform
import sys
import warnings
from os.path import dirname, join

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QWidget

from . import __version__
from ._qt.qt_main_window import Window
from ._qt.qt_viewer import QtViewer
from ._qt.qthreading import wait_for_workers_to_quit
from .components import ViewerModel
from .utils import config
from .utils.perf import perf_config


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string, optional
        The title of the viewer window. by default 'napari'.
    ndisplay : {2, 3}, optional
        Number of displayed dimensions. by default 2.
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. by default None
    axis_labels : list of str, optional
        Dimension names. by default they are labeled with sequential numbers
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.
    """

    # set _napari_app_id to False to avoid overwriting dock icon on windows
    # set _napari_app_id to custom string to prevent grouping different base viewer
    _napari_app_id = 'napari.napari.viewer.' + str(__version__)

    def __init__(
        self,
        *,
        title='napari',
        ndisplay=2,
        order=None,
        axis_labels=None,
        show=True,
    ):
        # instance() returns the singleton instance if it exists, or None
        app = QApplication.instance()
        # if None, raise a RuntimeError with the appropriate message
        if app is None:
            message = (
                "napari requires a Qt event loop to run. To create one, "
                "try one of the following: \n"
                "  - use the `napari.gui_qt()` context manager. See "
                "https://github.com/napari/napari/tree/master/examples for"
                " usage examples.\n"
                "  - In IPython or a local Jupyter instance, use the "
                "`%gui qt` magic command.\n"
                "  - Launch IPython with the option `--gui=qt`.\n"
                "  - (recommended) in your IPython configuration file, add"
                " or uncomment the line `c.TerminalIPythonApp.gui = 'qt'`."
                " Then, restart IPython."
            )
            raise RuntimeError(message)

        if perf_config:
            if perf_config.trace_qt_events:
                from ._qt.tracing.qt_event_tracing import (
                    convert_app_for_tracing,
                )

                # For tracing Qt events we need a special QApplication. If
                # using `gui_qt` we already have the special one, and no
                # conversion is done here. However when running inside
                # IPython or Jupyter this is where we switch out the
                # QApplication.
                app = convert_app_for_tracing(app)

            # Will patch based on config file.
            perf_config.patch_callables()

        if (
            platform.system() == "Windows"
            and not getattr(sys, 'frozen', False)
            and self._napari_app_id
        ):
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                self._napari_app_id
            )

        logopath = join(dirname(__file__), 'resources', 'logo.png')
        app.setWindowIcon(QIcon(logopath))

        # see docstring of `wait_for_workers_to_quit` for caveats on killing
        # workers at shutdown.
        app.aboutToQuit.connect(wait_for_workers_to_quit)

        super().__init__(
            title=title,
            ndisplay=ndisplay,
            order=order,
            axis_labels=axis_labels,
        )
        qt_viewer = QtViewer(self)
        self._window = Window(qt_viewer, show=show)

    @property
    def window(self):
        warnings.warn(
            (
                "The viewer.window parameter is deprecated and will be removed in version 0.4.2."
                " Instead you should use the viewer.add_dock_widget method to extend the GUI."
                " Directly manipulation of the window is no longer supported."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._window

    def update_console(self, variables):
        """Update console's namespace with desired variables.

        Parameters
        ----------
        variables : dict, str or list/tuple of str
            The variables to inject into the console's namespace.  If a dict, a
            simple update is done.  If a str, the string is assumed to have
            variable names separated by spaces.  A list/tuple of str can also
            be used to give the variable names.  If just the variable names are
            give (list/tuple/str) then the variable values looked up in the
            callers frame.
        """
        if self._window.qt_viewer.console is None:
            return
        else:
            self._window.qt_viewer.console.push(variables)

    def screenshot(self, path=None, *, canvas_only=True):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
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
        if canvas_only:
            image = self._window.qt_viewer.screenshot(path=path)
        else:
            image = self._window.screenshot(path=path)
        return image

    def add_dock_widget(
        self,
        widget: QWidget,
        *,
        name: str = '',
        area: str = 'bottom',
        allowed_areas=None,
        shortcut=None,
    ):
        """Add a Dock Widget to the main window to extend the GUI functionality.

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
        return self._window.add_dock_widget(
            widget=widget,
            name=name,
            area=area,
            allowed_areas=allowed_areas,
            shortcut=shortcut,
        )

    def show(self):
        """Resize, show, and raise the viewer window."""
        self._window.show()

    def close(self):
        """Close the viewer window."""
        self._window.close()

        if config.async_loading:
            from .components.experimental.chunk import chunk_loader

            # TODO_ASYNC: Find a cleaner way to do this? Fixes some tests.
            # https://github.com/napari/napari/issues/1500
            for layer in self.layers:
                chunk_loader.on_layer_deleted(layer)

    def __str__(self):
        """Simple string representation"""
        return f'napari.Viewer: {self.title}'

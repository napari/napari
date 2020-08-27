import platform
import sys
from os.path import dirname, join

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from . import __version__
from ._qt.qt_main_window import Window
from ._qt.qt_viewer import QtViewer
from ._qt.qthreading import create_worker, wait_for_workers_to_quit
from .components import ViewerModel
from .components.chunk import chunk_loader
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
        self.window = Window(qt_viewer, show=show)

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
        if self.window.qt_viewer.console is None:
            return
        else:
            self.window.qt_viewer.console.push(variables)

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
            image = self.window.qt_viewer.screenshot(path=path)
        else:
            image = self.window.screenshot(path=path)
        return image

    def update(self, func, *args, **kwargs):
        import warnings

        warnings.warn(
            "Viewer.update() is deprecated, use "
            "create_worker(func, *args, **kwargs) instead",
            DeprecationWarning,
        )
        return create_worker(func, *args, **kwargs, _start_thread=True)

    def show(self):
        """Resize, show, and raise the viewer window."""
        self.window.show()

    def close(self):
        """Close the viewer window."""
        self.window.close()

        # TODO_ASYNC: Tell the ChunkLoader which layers are in the
        # viewer that's being closed. This is surely not what we want
        # to do long term, but it fixes some tests for now. See:
        # https://github.com/napari/napari/issues/1500
        for layer in self.layers:
            chunk_loader.on_layer_deleted(layer)

    def __str__(self):
        """Simple string representation"""
        return f'napari.Viewer: {self.title}'

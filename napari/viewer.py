from os.path import dirname, join

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication

from napari._qt.qt_update_ui import QtUpdateUI

from ._qt.qt_main_window import Window
from ._qt.qt_viewer import QtViewer
from .components import ViewerModel
from ._dask import DaskRemoteViewer


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str
        Dimension names.
    """

    def __init__(
        self,
        title='napari',
        ndisplay=2,
        order=None,
        axis_labels=None,
        is_remote=False,
        address=None,
    ):
        self.is_remote = is_remote
        self.address = address

        if self.is_remote is False:
            # If not remote create and connect to a window
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

            logopath = join(dirname(__file__), 'resources', 'logo.png')
            app.setWindowIcon(QIcon(logopath))

            super().__init__(title=title, ndisplay=ndisplay, order=order)

            qt_viewer = QtViewer(self)
            self.window = Window(qt_viewer)
            self.screenshot = self.window.qt_viewer.screenshot
            self.update_console = self.window.qt_viewer.console.push
        else:
            super().__init__(title=title, ndisplay=ndisplay, order=order)
            app = None

        if self.address is not None:
            self.remote = DaskRemoteViewer(self, self.address, qapp=app)

    def screenshot(self, with_viewer=False):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        with_viewer : bool
            If True includes the napari viewer, otherwise just includes the
            canvas.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        if self.is_remote:
            return None

        if with_viewer:
            image = self.window.screenshot()
        else:
            image = self.window.qt_viewer.screenshot()
        return image

    def update(self, func, *args, **kwargs):
        if self.is_remote:
            return None

        t = QtUpdateUI(func, *args, **kwargs)
        self.window.qt_viewer.pool.start(t)
        return self.window.qt_viewer.pool  # returns threadpool object

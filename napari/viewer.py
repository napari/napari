from os.path import dirname, join
import sys

from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen

from ._qt.qt_update_ui import QtUpdateUI
from ._qt.qt_main_window import Window
from ._qt.qt_viewer import QtViewer
from .components import ViewerModel


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
        self, title='napari', ndisplay=2, order=None, axis_labels=None, startup_logo=False,
    ):
        # instance() returns the singleton instance if it exists, or None
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)
            self._app.setApplicationName('napari')

        logopath = join(dirname(__file__), 'resources', 'logo.png')
        self._app.setWindowIcon(QIcon(logopath))

        super().__init__(
            title=title,
            ndisplay=ndisplay,
            order=order,
            axis_labels=axis_labels,
        )
        self._startup_logo = startup_logo
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer)
        self.update_console = self.window.qt_viewer.console.push

    def __enter__(self):
        if self._startup_logo:
            self._splash_widget = QSplashScreen(QPixmap(logopath).scaled(400, 400))
            self._splash_widget.show()
        return self

    def __exit__(self, type, value, traceback):
        if self._startup_logo:
            self._splash_widget.close()
        self._app.exec_()

    def show(self):
        """Start the Qt event loop.
        Used when the viewer was instatiated outside of a
        context manager.
        """
        if self._app is None:
            return
        self._app.exec_()

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
        if with_viewer:
            image = self.window.screenshot()
        else:
            image = self.window.qt_viewer.screenshot()
        return image

    def update(self, func, *args, **kwargs):
        t = QtUpdateUI(func, *args, **kwargs)
        self.window.qt_viewer.pool.start(t)
        return self.window.qt_viewer.pool  # returns threadpool object

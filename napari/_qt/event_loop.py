import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen


@contextmanager
def gui_qt(splash_screen=True):
    """Start a Qt event loop in which to run the application.

    Parameters
    ----------
    splash_screen : bool
        Show a splash screen with the napari logo during startup.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the ``%gui qt`` magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    if splash_screen:
        logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
        splash_widget = QSplashScreen(QPixmap(logopath).scaled(400, 400))
        splash_widget.show()
    yield
    if splash_screen:
        splash_widget.close()
    app.exec_()

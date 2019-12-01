import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen


@contextmanager
def gui_qt():
    """Start a Qt event loop in which to run the application.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the ``%gui qt`` magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
    splash_screen = QSplashScreen(QPixmap(logopath).scaled(400, 400))
    splash_screen.show()
    yield
    splash_screen.close()
    app.exec_()

import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen


@contextmanager
def gui_qt(*, startup_logo=False):
    """Start a Qt event loop in which to run the application.

    Parameters
    ----------
    startup_logo : bool
        Show a splash screen with the napari logo during startup.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the ``%gui qt`` magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    app = QApplication.instance()
    if not app:
        # if this is the first time the Qt app is being instantiated, we set
        # the name, so that we know whether to raise_ in Window.show()
        app = QApplication(sys.argv)
        app.setApplicationName('napari')
    if startup_logo:
        logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
        splash_widget = QSplashScreen(QPixmap(logopath).scaled(400, 400))
        splash_widget.show()
    yield
    if startup_logo:
        splash_widget.close()
    app.exec_()

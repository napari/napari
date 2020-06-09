import os
import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen


def _create_application(argv) -> QApplication:
    """Create our QApplication.

    Notes
    -----

    We substitute QApplicationWithTiming when using perfmon.

    Note that in Viewer we call convert_app_for_timing() which will create a
    QApplicationWithTiming. However that's only for IPython/Jupyter. When using
    gui_qt we need to create it up front here before any QWidget objects are
    created, like the splash screen.
    """
    if os.getenv("NAPARI_PERFMON", "0") != "0":
        from .qt_event_timing import QApplicationWithTiming

        return QApplicationWithTiming(argv)
    else:
        return QApplication(argv)


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
    splash_widget = None
    app = QApplication.instance()
    if not app:
        # automatically determine monitor DPI.
        # Note: this MUST be set before the QApplication is instantiated
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        # if this is the first time the Qt app is being instantiated, we set
        # the name, so that we know whether to raise_ in Window.show()
        app = _create_application(sys.argv)
        app.setApplicationName('napari')
        if startup_logo:
            logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
            pm = QPixmap(logopath).scaled(
                360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            splash_widget = QSplashScreen(pm)
            splash_widget.show()
    yield app
    # if the application already existed before this function was called,
    # there's no need to start it again.  By avoiding unnecessary calls to
    # ``app.exec_``, we avoid blocking.
    if app.applicationName() == 'napari':
        if splash_widget and startup_logo:
            splash_widget.close()
        app.exec_()

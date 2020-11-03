import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen

from ..utils.perf import perf_config
from .exceptions import ExceptionHandler


def _create_application(argv) -> QApplication:
    """Create our QApplication.

    Notes
    -----
    Substitute QApplicationWithTracing when using perfmon.

    With IPython/Jupyter we call convert_app_for_tracing() which deletes
    the QApplication and creates a new one. However here with gui_qt we
    need to create the correct QApplication up front, or we will crash.
    We'll crash because we'd be deleting the QApplication after we created
    QWidgets with it, such as we do for the splash screen.
    """
    if perf_config and perf_config.trace_qt_events:
        from .tracing.qt_event_tracing import QApplicationWithTracing

        return QApplicationWithTracing(argv)
    else:
        return QApplication(argv)


@contextmanager
def gui_qt(*, startup_logo=False, gui_exceptions=False):
    """Start a Qt event loop in which to run the application.

    Parameters
    ----------
    startup_logo : bool, optional
        Show a splash screen with the napari logo during startup.
    gui_exceptions : bool, optional
        Whether to show uncaught exceptions in the GUI, by default they will be
        shown in the console that launched the event loop.

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
            app._splash_widget = splash_widget
    else:
        app._existed = True

    # instantiate the exception handler
    exception_handler = ExceptionHandler(gui_exceptions=gui_exceptions)
    sys.excepthook = exception_handler.handle

    try:
        yield app
    except Exception:
        exception_handler.handle(*sys.exc_info())

    # if the application already existed before this function was called,
    # there's no need to start it again.  By avoiding unnecessary calls to
    # ``app.exec_``, we avoid blocking.
    if app.applicationName() == 'napari':
        if splash_widget and startup_logo:
            splash_widget.close()
        app.exec_()

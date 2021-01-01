import sys
from contextlib import contextmanager
from os.path import dirname, join

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen

from ..utils.perf import perf_config
from .exceptions import ExceptionHandler
from .qthreading import wait_for_workers_to_quit


def get_app() -> QApplication:
    """Get or create the Qt QApplication

    Notes
    -----
    Substitute QApplicationWithTracing when using perfmon.

    With IPython/Jupyter we call convert_app_for_tracing() which deletes
    the QApplication and creates a new one. However here with gui_qt we
    need to create the correct QApplication up front, or we will crash because
    we'd be deleting the QApplication after we created QWidgets with it,
    such as we do for the splash screen.
    """
    app = QApplication.instance()
    if app:
        if perf_config and perf_config.trace_qt_events:
            from .tracing.qt_event_tracing import convert_app_for_tracing

            # no-op if app is already a QApplicationWithTracing
            app = convert_app_for_tracing(app)
        app._existed = True
    else:
        # automatically determine monitor DPI.
        # Note: this MUST be set before the QApplication is instantiated
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

        if perf_config and perf_config.trace_qt_events:
            from .tracing.qt_event_tracing import QApplicationWithTracing

            app = QApplicationWithTracing(sys.argv)
        else:
            app = QApplication(sys.argv)

        # if this is the first time the Qt app is being instantiated, we set
        # the name, so that we know whether to raise_ in Window.show()
        from napari import __version__

        app.setApplicationName('napari')
        app.setApplicationVersion(__version__)
        app.setOrganizationName('napari')
        app.setOrganizationDomain('napari.org')

    if perf_config and not perf_config.patched:
        # Will patch based on config file.
        perf_config.patch_callables()

    # see docstring of `wait_for_workers_to_quit` for caveats on killing
    # workers at shutdown.
    app.aboutToQuit.connect(wait_for_workers_to_quit)

    return app


@contextmanager
def gui_qt(*, startup_logo=False, gui_exceptions=False, force=False):
    """Start a Qt event loop in which to run the application.

    Parameters
    ----------
    startup_logo : bool, optional
        Show a splash screen with the napari logo during startup.
    gui_exceptions : bool, optional
        Whether to show uncaught exceptions in the GUI, by default they will be
        shown in the console that launched the event loop.
    force : bool, optional
        Force the application event_loop to start, even if there are no top
        level widgets to show.

    Notes
    -----
    This context manager is not needed if running napari within an interactive
    IPython session. In this case, use the ``%gui qt`` magic command, or start
    IPython with the Qt GUI event loop enabled by default by using
    ``ipython --gui=qt``.
    """
    splash_widget = None

    app = get_app()
    if startup_logo and app.applicationName() == 'napari':
        logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
        pm = QPixmap(logopath).scaled(
            360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        splash_widget = QSplashScreen(pm)
        splash_widget.show()
        app._splash_widget = splash_widget

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
    # we add 'magicgui' so that anyone using @magicgui *before* calling gui_qt
    # will also have the application executed. (a bandaid for now?...)
    # see https://github.com/napari/napari/pull/2016
    if app.applicationName() in ('napari', 'magicgui'):
        if splash_widget and startup_logo:
            splash_widget.close()
        run_app(force=force, _func_name='gui_qt')


def run_app(*, force=False, _func_name='run_app'):
    """Start the Qt Event Loop

    Parameters
    ----------
    force : bool, optional
        Force the application event_loop to start, even if there are no top
        level widgets to show.
    _func_name : str, optional
        name of calling function, by default 'run_app'.  This is only here to
        provide functions like `gui_qt` a way to inject their name into the
        warning message.

    Raises
    ------
    RuntimeError
        (To avoid confusion) if no widgets would be shown upon starting the
        event loop.
    """
    app = QApplication.instance()
    if not app:
        raise RuntimeError(
            'No Qt app has been created. '
            'One can be created by calling `get_app()` '
            'or qtpy.QtWidgets.QApplication([])'
        )
    if not app.topLevelWidgets() and not force:
        from warnings import warn

        warn(
            "Refusing to run a QApplication with no topLevelWidgets. "
            f"To run the app anyway, use `{_func_name}(force=True)`"
        )
        return
    app.exec_()

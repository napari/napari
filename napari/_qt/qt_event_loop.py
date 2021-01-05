import os
import sys
from contextlib import contextmanager

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtWidgets import QApplication, QSplashScreen

from napari import __version__

from ..utils.perf import perf_config
from .exceptions import ExceptionHandler
from .qthreading import wait_for_workers_to_quit

NAPARI_ICON_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'resources', 'logo.png'
)
NAPARI_APP_ID = f'napari.napari.viewer.{__version__}'


def set_app_id(app_id):
    if os.name == "nt" and app_id and not getattr(sys, 'frozen', False):
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


_defaults = {
    'app_name': 'napari',
    'app_version': __version__,
    'icon': NAPARI_ICON_PATH,
    'org_name': 'napari',
    'org_domain': 'napari.org',
    'app_id': NAPARI_APP_ID,
}


def get_app(
    *,
    app_name: str = None,
    app_version: str = None,
    icon: str = None,
    org_name: str = None,
    org_domain: str = None,
    app_id: str = None,
) -> QApplication:
    """Get or create the Qt QApplication.

    There is only one global QApplication instance, which can be retrieved by
    calling get_app again, (or by using QApplication.instance())

    Parameters
    ----------
    app_name : str, optional
        Set app name (if creating for the first time), by default 'napari'
    app_version : str, optional
        Set app version (if creating for the first time), by default __version__
    icon : str, optional
        Set app icon (if creating for the first time), by default
        NAPARI_ICON_PATH
    org_name : str, optional
        Set organization name (if creating for the first time), by default
        'napari'
    org_domain : str, optional
        Set organization domain (if creating for the first time), by default
        'napari.org'
    app_id : str, optional
        Set organization domain (if creating for the first time).  Will be
        passed to set_app_id (which may also be called independently), by
        default NAPARI_APP_ID

    Returns
    -------
    QApplication
        [description]

    Notes
    -----
    Substitutes QApplicationWithTracing when the NAPARI_PERFMON env variable
    is set.

    If the QApplication already exists, we call convert_app_for_tracing() which
    deletes the QApplication and creates a new one. However here with get_app
    we need to create the correct QApplication up front, or we will crash
    because we'd be deleting the QApplication after we created QWidgets with
    it, such as we do for the splash screen.
    """
    # napari defaults are all-or nothing.  If any of the keywords are used
    # then they are all used.
    set_values = {k for k, v in locals().items() if v}
    kwargs = locals() if set_values else _defaults

    app = QApplication.instance()
    if app:
        if set_values:
            from warnings import warn

            warn(
                "QApplication already existed, these arguments to to 'get_app'"
                " were ignored: {}".format(set_values)
            )
        if perf_config and perf_config.trace_qt_events:
            from .perf.qt_event_tracing import convert_app_for_tracing

            # no-op if app is already a QApplicationWithTracing
            app = convert_app_for_tracing(app)
        app._existed = True
    else:
        # automatically determine monitor DPI.
        # Note: this MUST be set before the QApplication is instantiated
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

        if perf_config and perf_config.trace_qt_events:
            from .perf.qt_event_tracing import QApplicationWithTracing

            app = QApplicationWithTracing(sys.argv)
        else:
            app = QApplication(sys.argv)

        # if this is the first time the Qt app is being instantiated, we set
        # the name, so that we know whether to raise_ in Window.show()

        app.setApplicationName(kwargs.get('app_name'))
        app.setApplicationVersion(kwargs.get('app_version'))
        app.setOrganizationName(kwargs.get('org_name'))
        app.setOrganizationDomain(kwargs.get('org_domain'))
        app.setWindowIcon(QIcon(kwargs.get('icon')))
        set_app_id(kwargs.get('app_id'))

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
        pm = QPixmap(NAPARI_ICON_PATH).scaled(
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
        run(force=force, _func_name='gui_qt')


def run(*, force=False, _func_name='run'):
    """Start the Qt Event Loop

    Parameters
    ----------
    force : bool, optional
        Force the application event_loop to start, even if there are no top
        level widgets to show.
    _func_name : str, optional
        name of calling function, by default 'run'.  This is only here to
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

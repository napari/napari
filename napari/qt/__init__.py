import warnings

from .._qt.qt_event_loop import get_app, run
from .._qt.qt_main_window import Window
from .._qt.qt_resources import compile_qt_svgs, get_stylesheet
from .._qt.qt_viewer import QtViewer
from .._qt.widgets.qt_viewer_buttons import QtStateButton, QtViewerButtons
from ..utils.translations import trans
from .progress import progrange, progress
from .threading import create_worker, thread_worker


class QtNDisplayButton(QtStateButton):
    def __init__(self, viewer):
        warnings.warn(
            trans._(
                "QtNDisplayButton is deprecated and will be removed in 0.4.9. Instead a more general QtStateButton is provided."
            ),
            stacklevel=2,
        )
        super().__init__(
            "ndisplay_button",
            viewer.dims,
            'ndisplay',
            viewer.dims.events.ndisplay,
            2,
            3,
        )


__all__ = (
    'compile_qt_svgs',
    'create_worker',
    'progress',
    'progrange',
    'QtStateButton',
    'QtNDisplayButton',
    'QtViewer',
    'QtViewerButtons',
    'thread_worker',
    'Window',
    'get_app',
    'get_stylesheet',
    'run',
)

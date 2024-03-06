"""Qt providers.

Non-Qt providers can be found in `napari/_app_model/injection/_providers.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window
    from napari._qt.qt_viewer import QtViewer


def _provide_qt_viewer() -> Optional[QtViewer]:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._qt_viewer
    return None


def _provide_qt_viewer_or_raise(msg: str = '') -> QtViewer:
    qt_viewer = _provide_qt_viewer()
    if qt_viewer:
        return qt_viewer
    if msg:
        msg = ' ' + msg
    raise RuntimeError(
        trans._(
            'No current `QtViewer` found.{msg}',
            deferred=True,
            msg=msg,
        )
    )


def _provide_window() -> Optional[Window]:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._window
    return None


def _provide_window_or_raise(msg: str = '') -> Window:
    window = _provide_window()
    if window:
        return window
    if msg:
        msg = ' ' + msg
    raise RuntimeError(
        trans._(
            'No current `Window` found.{msg}',
            deferred=True,
            msg=msg,
        )
    )


QPROVIDERS = [
    (_provide_qt_viewer,),
    (_provide_window,),
]

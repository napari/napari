"""Qt providers.

Non-Qt providers can be found in `napari/_app_model/injection/_providers.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window
    from napari._qt.qt_viewer import QtViewer


def _provide_qt_viewer() -> Optional[QtViewer]:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._qt_viewer
    return None


def _provide_window() -> Optional[Window]:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._window
    return None


QPROVIDERS = [
    (_provide_qt_viewer,),
    (_provide_window,),
]

"""Qt providers.

Non-Qt providers can be found in `napari/_app_model/injection/_providers.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window
    from napari._qt.qt_viewer import QtViewer


def _provide_qt_viewer(
    raise_error: Union[bool, str] = False
) -> Optional[QtViewer]:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._qt_viewer
    if raise_error:
        msg = ''
        if isinstance(raise_error, str):
            msg = ' ' + raise_error
        raise RuntimeError(  # pragma: no cover
            trans._('No current `Viewer` found.{msg}', deferred=True, msg=msg)
        )
    return None


def _provide_window(raise_error: Union[bool, str] = False) -> Optional[Window]:
    from napari._qt.qt_main_window import _QtMainWindow

    if _qmainwin := _QtMainWindow.current():
        return _qmainwin._window
    if raise_error:
        msg = ''
        if isinstance(raise_error, str):
            msg = ' ' + raise_error
        raise RuntimeError(  # pragma: no cover
            trans._('No current `Window` found.{msg}', deferred=True, msg=msg)
        )
    return None


QPROVIDERS = [
    (_provide_qt_viewer,),
    (_provide_window,),
]

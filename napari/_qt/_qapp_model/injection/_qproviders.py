"""Qt providers.

Non-Qt providers can be found in `napari/_app_model/injection/_providers.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from magicgui.widgets import create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton

from napari._qt.dialogs.qt_modal import QtPopup
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari._qt.qt_main_window import Window
    from napari._qt.qt_viewer import QtViewer
    from napari.layers import SourceLayer


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


class LayerPopup(QtPopup):
    def __init__(self, qt_viewer: QtViewer):
        super().__init__(qt_viewer.layers)
        self.selected_layer = None

        self.setObjectName('LayerPopup')
        self.select = QPushButton('Select')
        self.select_combo = create_widget(annotation='napari.layers.Layer')
        self.select.clicked.connect(self._accept)

        self._layput = QHBoxLayout()
        self._layput.setContentsMargins(10, 16, 10, 16)
        self._layput.addWidget(self.select_combo.native)
        self._layput.addWidget(self.select)
        self.frame.setLayout(self._layput)

    def _accept(self):
        self.selected_layer = self.select_combo.value
        self.close()


def _provide_source_layer() -> Optional[SourceLayer]:
    viewer = _provide_qt_viewer()

    if viewer is None:
        return None
    pop = LayerPopup(viewer)
    pop.exec_()
    return pop.selected_layer


QPROVIDERS = [
    (_provide_qt_viewer,),
    (_provide_window,),
    (_provide_source_layer,),
]

"""Status bar widget on the viewer MainWindow"""
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QMouseEvent, QMovie
from qtpy.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

import napari.resources

from ...utils.translations import trans
from ..dialogs.qt_activity_dialog import ActivityToggleItem

if TYPE_CHECKING:
    from ..qt_main_window import _QtMainWindow


class UpdateStatus(QWidget):

    clicked = Signal()

    def __init__(self, parent=None, text=''):
        super().__init__(parent=parent)
        self._movie = QLabel('')
        self._text = QLabel(text)

        # Setup
        self._movie = QLabel(trans._("updating ..."), self)
        sp = self._movie.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self._movie.setSizePolicy(sp)
        load_gif = str(Path(napari.resources.__file__).parent / "loading.gif")
        mov = QMovie(load_gif)
        mov.setScaledSize(QSize(12, 12))
        self._movie.setMovie(mov)
        mov.start()
        self._movie.setVisible(False)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self._movie)
        layout.addWidget(self._text)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)
        self._movie

    def setText(self, text):
        self._movie.setVisible(bool(text))
        self._text.setText(text)

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        if self._text.text():
            self.clicked.emit()

        return super().mousePressEvent(a0)


class ViewerStatusBar(QStatusBar):
    def __init__(self, parent: '_QtMainWindow') -> None:
        super().__init__(parent=parent)

        self.showMessage(trans._('Ready'))
        self._help = QLabel('')
        self.addPermanentWidget(self._help)
        self._update_status = UpdateStatus(self)
        self.addPermanentWidget(self._update_status)
        self._update_status.setText("")
        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        # FIXME: feels weird to set this here.
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)

    def setHelpText(self, text: str) -> None:
        self._help.setText(text)

    def setUpdateStatus(self, value) -> None:
        self._update_status.setText(value)

    def _toggle_activity_dock(self, visible: bool):
        par: _QtMainWindow = self.parent()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(Qt.DownArrow)
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.UpArrow)

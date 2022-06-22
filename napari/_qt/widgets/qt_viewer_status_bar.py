"""Status bar widget on the viewer MainWindow"""
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QStatusBar
from superqt import QElidingLabel

from ...utils.translations import trans
from ..dialogs.qt_activity_dialog import ActivityToggleItem

if TYPE_CHECKING:
    from ..qt_main_window import _QtMainWindow

STATUS_MSG_WIDTH = 600


class ViewerStatusBar(QStatusBar):
    def __init__(self, parent: '_QtMainWindow') -> None:
        super().__init__(parent=parent)

        self._status_message = QElidingLabel(trans._('Ready'))
        self._status_message.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.addWidget(self._status_message)
        self._help = QLabel('')
        self.addPermanentWidget(self._help)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(
            self._toggle_activity_dock
        )
        # FIXME: feels weird to set this here.
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)

    def setHelpText(self, text: str) -> None:
        self._help.setText(text)

    def setStatusText(self, text: str) -> None:
        self._status_message.resize(
            STATUS_MSG_WIDTH, self._status_message.height()
        )
        self._status_message.setText(text)

    def _toggle_activity_dock(self, visible: bool):
        par: _QtMainWindow = self.parent()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(Qt.DownArrow)
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.UpArrow)

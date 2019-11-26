from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QDialog,
    QFormLayout,
    QFrame,
)
from qtpy.QtGui import QCursor


class QtModalPopup(QDialog):
    """A generic modal popup window.

    The seemingly extra frame here is to allow rounded corners on a truly
    transparent background

    +-------------------------------
    | Dialog
    |  +----------------------------
    |  | QVBoxLayout
    |  |  +-------------------------
    |  |  | QFrame
    |  |  |  +----------------------
    |  |  |  | QFormLayout
    |  |  |  |
    |  |  |  |  (stuff goes here)
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("QtModalPopup")
        self.setModal(True)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setLayout(QVBoxLayout())

        self.frame = QFrame()
        self.layout().addWidget(self.frame)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.form_layout = QFormLayout()
        closebutton = QPushButton("Close")
        closebutton.clicked.connect(self.close)
        self.form_layout.addRow(closebutton)
        self.frame.setLayout(self.form_layout)

    def show_above_mouse(self, *args):
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(szhint.width() / 2, szhint.height() + 14)
        self.move(pos)
        self.show()

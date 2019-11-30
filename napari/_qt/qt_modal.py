from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QVBoxLayout, QDialog, QFormLayout, QFrame
from qtpy.QtGui import QCursor


class QtPopup(QDialog):
    """A generic popup window.

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
        self.setModal(False)  # if False, then clicking anywhere else closes it
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setLayout(QVBoxLayout())

        self.frame = QFrame()
        self.frame.setObjectName("QtPopupFrame")
        self.layout().addWidget(self.frame)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.form_layout = QFormLayout()
        self.frame.setLayout(self.form_layout)

    def show_above_mouse(self, *args):
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(szhint.width() / 2, szhint.height() + 14)
        self.move(pos)
        self.show()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            return self.close()
        super().keyPressEvent(event)

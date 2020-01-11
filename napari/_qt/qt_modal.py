from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QVBoxLayout, QDialog, QFrame
from qtpy.QtGui import QCursor
from .utils import find_ancestor_mainwindow


class QtPopup(QDialog):
    """A generic popup window.

    The seemingly extra frame here is to allow rounded corners on a truly
    transparent background.  New items should be added to QtPopup.frame

    +----------------------------------
    | Dialog
    |  +-------------------------------
    |  | QVBoxLayout
    |  |  +----------------------------
    |  |  | QFrame
    |  |  |  +-------------------------
    |  |  |  |
    |  |  |  |  (add a new layout here)
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

    def show_above_mouse(self, *args):
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(szhint.width() / 2, szhint.height() + 14)
        self.move(pos)
        self.show()

    def show_at(self, position='top', *, width_ratio=0.9):
        """Show popup at a position relative to the QMainWindow.

        Parameters
        ----------
        position : str, {'top', 'bottom', 'left', 'right' } optional
            position in the QMainWindow to show the pop, by default 'top'
        width_ratio : float, optional
            Fraction of the width (for position = top/bottom) or height (for
            position = left/right) of the QMainWindow that the popup will
            occupy,  by default 0.9

        Raises
        ------
        NotImplementedError
            if the QtPopup does not have a parent (not clear how to find main
            window.)
        ValueError
            if position is not one of {'top', 'bottom', 'left', 'right' }
        """
        if not self.parent():
            raise NotImplementedError("cannot use `show_at` without parent")
        main_window = find_ancestor_mainwindow(self.parent())
        xy = main_window.pos()
        if position == 'top':
            width = main_window.width() * width_ratio
            height = self.sizeHint().height()
            xy = xy + QPoint(main_window.width() * (1 - width_ratio) / 2, 24)
        elif position == 'bottom':
            width = main_window.width() * width_ratio
            height = self.sizeHint().height()
            y = main_window.height() - self.height() - 2
            xy = xy + QPoint(main_window.width() * (1 - width_ratio) / 2, y)
        elif position == 'left':
            width = self.sizeHint().width()
            height = main_window.height() * width_ratio
            xy = xy + QPoint(12, main_window.height() * (1 - width_ratio) / 2)
        elif position == 'right':
            width = self.sizeHint().width()
            height = main_window.height() * width_ratio
            x = main_window.width() - width - 12
            xy = xy + QPoint(x, main_window.height() * (1 - width_ratio) / 2)
        else:
            raise ValueError(
                'position must be one of ["top", "left", "bottom", "right"]'
            )

        # necessary for transparent round corners
        self.resize(self.sizeHint())
        self.setGeometry(xy.x(), xy.y(), max(width, 20), max(height, 20))
        self.show()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            return self.close()
        super().keyPressEvent(event)

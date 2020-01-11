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
        position : {str, tuple}, optional
            position in the QMainWindow to show the pop, by default 'top'
            if str: must be one of {'top', 'bottom', 'left', 'right' }
            if tuple: must be length 4 with (left, top, width, height)
        width_ratio : float, optional
            Fraction of the width (for position = top/bottom) or height (for
            position = left/right) of the QMainWindow that the popup will
            occupy.  Only valid when isinstance(position, str). 
            by default 0.9

        Raises
        ------
        ValueError
            if position is a string and not one of
            {'top', 'bottom', 'left', 'right' }
        """
        if isinstance(position, str):
            main_window = find_ancestor_mainwindow(self)
            if main_window:
                xy = main_window.pos()
                width = main_window.width()
                height = main_window.height()
            else:
                # fallback... at least show something.  This partially to make
                # testing easier
                xy = QPoint(200, 200)
                width = 600
                height = 60
            if position == 'top':
                width = width * width_ratio
                height = self.sizeHint().height()
                xy = xy + QPoint(width * (1 - width_ratio) / 2, 24)
            elif position == 'bottom':
                width = width * width_ratio
                height = self.sizeHint().height()
                y = height - self.height() - 2
                xy = xy + QPoint(width * (1 - width_ratio) / 2, y)
            elif position == 'left':
                width = self.sizeHint().width()
                height = height * width_ratio
                xy = xy + QPoint(12, height * (1 - width_ratio) / 2)
            elif position == 'right':
                width = self.sizeHint().width()
                height = height * width_ratio
                x = width - width - 12
                xy = xy + QPoint(x, height * (1 - width_ratio) / 2)
            else:
                raise ValueError(
                    'position must be one of '
                    '["top", "left", "bottom", "right"]'
                )
        elif isinstance(position, (tuple, list)):
            assert len(position) == 4, '`position` argument must have length 4'
            x, y, width, height = position
            xy = QPoint(x, y)

        # necessary for transparent round corners
        self.resize(self.sizeHint())
        self.setGeometry(xy.x(), xy.y(), max(width, 20), max(height, 20))
        self.show()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            return self.close()
        super().keyPressEvent(event)

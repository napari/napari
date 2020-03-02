from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QCursor, QGuiApplication
from qtpy.QtWidgets import QDialog, QFrame, QVBoxLayout


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

    def show_right_of_mouse(self, *args):
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(-14, szhint.height() / 4)
        self.move(pos)
        self.show()

    def show_at(self, position='top', *, win_ratio=0.9, min_length=0):
        """Show popup at a position relative to the QMainWindow.

        Parameters
        ----------
        position : {str, tuple}, optional
            position in the QMainWindow to show the pop, by default 'top'
            if str: must be one of {'top', 'bottom', 'left', 'right' }
            if tuple: must be length 4 with (left, top, width, height)
        win_ratio : float, optional
            Fraction of the width (for position = top/bottom) or height (for
            position = left/right) of the QMainWindow that the popup will
            occupy.  Only valid when isinstance(position, str).
            by default 0.9
        min_length : int, optional
            Minimum size of the long dimension (width for top/bottom or
            height fort left/right).

        Raises
        ------
        ValueError
            if position is a string and not one of
            {'top', 'bottom', 'left', 'right' }
        """
        if isinstance(position, str):
            window = self.parent().window() if self.parent() else None
            if not window:
                raise ValueError(
                    "Specifying position as a string is only posible if "
                    "the popup has a parent"
                )
            left = window.pos().x()
            top = window.pos().y()
            if position in ('top', 'bottom'):
                width = window.width() * win_ratio
                width = max(width, min_length)
                left += (window.width() - width) / 2
                height = self.sizeHint().height()
                top += (
                    24
                    if position == 'top'
                    else (window.height() - height - 12)
                )
            elif position in ('left', 'right'):
                height = window.height() * win_ratio
                height = max(height, min_length)
                # 22 is for the title bar
                top += 22 + (window.height() - height) / 2
                width = self.sizeHint().width()
                left += (
                    12 if position == 'left' else (window.width() - width - 12)
                )
            else:
                raise ValueError(
                    'position must be one of '
                    '["top", "left", "bottom", "right"]'
                )
        elif isinstance(position, (tuple, list)):
            assert len(position) == 4, '`position` argument must have length 4'
            left, top, width, height = position

        # necessary for transparent round corners
        self.resize(self.sizeHint())
        # make sure the popup is completely on the screen
        screen_size = QGuiApplication.screenAt(QCursor.pos()).size()
        left = max(min(screen_size.width() - width, left), 0)
        top = max(min(screen_size.height() - height, top), 0)
        self.setGeometry(left, top, width, height)
        self.show()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            return self.close()
        super().keyPressEvent(event)

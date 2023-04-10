from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtGui import QCursor, QGuiApplication
from qtpy.QtWidgets import QDialog, QFrame, QVBoxLayout

from napari.utils.translations import trans


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

    Parameters
    ----------
    parent : qtpy.QtWidgets:QWidget
        Parent widget of the popup dialog box.

    Attributes
    ----------
    frame : qtpy.QtWidgets.QFrame
        Frame of the popup dialog box.
    """

    def __init__(self, parent) -> None:
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
        """Show popup dialog above the mouse cursor position."""
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(szhint.width() // 2, szhint.height() + 14)
        self.move(pos)
        self.show()

    def show_right_of_mouse(self, *args):
        pos = QCursor().pos()  # mouse position
        szhint = self.sizeHint()
        pos -= QPoint(-14, szhint.height() // 4)
        self.move(pos)
        self.show()

    def move_to(self, position='top', *, win_ratio=0.9, min_length=0):
        """Move popup to a position relative to the QMainWindow.

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
                    trans._(
                        "Specifying position as a string is only possible if the popup has a parent",
                        deferred=True,
                    )
                )
            left = window.pos().x()
            top = window.pos().y()
            if position in ('top', 'bottom'):
                width = int(window.width() * win_ratio)
                width = max(width, min_length)
                left += (window.width() - width) // 2
                height = self.sizeHint().height()
                top += (
                    24
                    if position == 'top'
                    else (window.height() - height - 12)
                )
            elif position in ('left', 'right'):
                height = int(window.height() * win_ratio)
                height = max(height, min_length)
                # 22 is for the title bar
                top += 22 + (window.height() - height) // 2
                width = self.sizeHint().width()
                left += (
                    12 if position == 'left' else (window.width() - width - 12)
                )
            else:
                raise ValueError(
                    trans._(
                        'position must be one of ["top", "left", "bottom", "right"]',
                        deferred=True,
                    )
                )
        elif isinstance(position, (tuple, list)):
            assert len(position) == 4, '`position` argument must have length 4'
            left, top, width, height = position
        else:
            raise TypeError(
                trans._(
                    "Wrong type of position {position}",
                    deferred=True,
                    position=position,
                )
            )

        # necessary for transparent round corners
        self.resize(self.sizeHint())
        # make sure the popup is completely on the screen
        # In Qt ≥5.10 we can use screenAt to know which monitor the mouse is on

        screen_geometry: QRect = QGuiApplication.screenAt(
            QCursor.pos()
        ).geometry()

        left = max(
            min(screen_geometry.right() - width, left), screen_geometry.left()
        )
        top = max(
            min(screen_geometry.bottom() - height, top), screen_geometry.top()
        )
        self.setGeometry(left, top, width, height)

    def keyPressEvent(self, event):
        """Close window on return, else pass event through to super class.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.close()
            return
        super().keyPressEvent(event)

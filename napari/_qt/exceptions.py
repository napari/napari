import logging
import traceback
from types import TracebackType
from typing import Type

from qtpy.QtCore import (
    Qt,
    QObject,
    Signal,
    QSize,
    QPoint,
    QPropertyAnimation,
    QEasingCurve,
    QTimer,
)
from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QMainWindow,
    QGraphicsOpacityEffect,
    QPushButton,
)


class ExceptionHandler(QObject):
    """General class to handle all uncaught exceptions in the Qt event loop"""

    error = Signal(tuple)
    message = None

    def handle(
        self,
        etype: Type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        """Our sys.excepthook override.

        This function handles uncaught exceptions and can delegate to a
        secondary handler, whether it be a GUI dialog, or an IPython traceback
        printout.  The override to ``sys.excepthook`` happens in
        :func:`napari.gui_qt`, and therefore this is only active when the qt
        event loop has been started by napari.

        The three parameters here are what would be returned from
        :func:`sys.exc_info()`.

        Parameters
        ----------
        etype : Type[BaseException]
            The type of error raised
        value : BaseException
            The error instance
        tb : TracebackType
            The traceback object associated with the error.
        """
        # etype.__module__ contains the module raising the error
        # Custom exception classes can have different behavior
        # can add custom exception handlers here ...
        text = "".join(traceback.format_exception(etype, value, tb))
        logging.error("Unhandled exception:\n%s", text)
        self._show_error_dialog(value)
        self.error.emit((etype, value, tb))

    def _show_error_dialog(self, exception: BaseException):
        self.message = NapariErrorMessage(exception)
        self.message.show()


class NapariErrorMessage(QMessageBox):
    def __init__(self, exception: BaseException):
        # FIXME: this works with command line, but not with IPython...
        # and may not work well with multiple viewers.
        parent = None
        for wdg in QApplication.topLevelWidgets():
            if isinstance(wdg, QMainWindow):
                parent = wdg
                break
        super().__init__(parent)

        # self.setIcon(QMessageBox.Warning)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setText(type(exception).__name__)
        self.setInformativeText(str(exception))
        self.setStandardButtons(QMessageBox.StandardButton.NoButton)
        self.opacity = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity)
        self.opacity_anim = QPropertyAnimation(self.opacity, b"opacity", self)
        self.geom_anim = QPropertyAnimation(self, b"geometry", self)
        self.setStyleSheet("QLabel{min-width: 250px;}")
        self.close_button = QPushButton(self)
        self.close_button.setObjectName("QErrorMessageCloseButton")
        self.close_button.clicked.connect(self.close)

    def show(self):
        """Show the message with a fade and slight slide in from the bottom.
        """
        # move to the bottom right
        sz = self.parent().size() - self.sizeHint() - QSize(22, 0)
        self.move(self.mapFromParent(QPoint(sz.width(), sz.height())))
        self.setFocusPolicy(Qt.NoFocus)
        # slide in
        geom = self.geometry()
        self.geom_anim.setDuration(220)
        self.geom_anim.setStartValue(geom)
        self.geom_anim.setEndValue(geom.translated(0, -50))
        self.geom_anim.setEasingCurve(QEasingCurve.OutQuad)
        # fade in
        self.opacity_anim.setDuration(200)
        self.opacity_anim.setStartValue(0)
        self.opacity_anim.setEndValue(1)
        self.geom_anim.start()
        self.opacity_anim.start()
        super().show()
        self.timer = QTimer()
        self.timer.singleShot(5000, self.close)

    def close(self):
        """Fade out then close."""
        self.opacity_anim.setDuration(80)
        self.opacity_anim.setStartValue(1)
        self.opacity_anim.setEndValue(0)
        self.opacity_anim.start()
        self.opacity_anim.finished.connect(super().close)

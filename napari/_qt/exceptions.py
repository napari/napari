import logging
import traceback
from types import TracebackType
from typing import Type

from qtpy.QtCore import Qt, QObject, Signal, QPoint
from qtpy.QtWidgets import QApplication, QErrorMessage, QMainWindow


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

    def _show_error_dialog(self, value: BaseException):
        if self.message is None:
            parent = None
            for wdg in QApplication.topLevelWidgets():
                if isinstance(wdg, QMainWindow):
                    parent = wdg
                    break
            self.message = QErrorMessage(parent)
            self.message.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
            self.message.setModal(False)
            self.message.move(self.message.mapToParent(QPoint(900, 600)))
            self.message.setFocusPolicy(Qt.NoFocus)

        self.message.showMessage(str(value))

import traceback
from types import TracebackType
from typing import Optional, Type

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

from ..types import ExcInfo
from ..utils.misc import camel_to_spaces


class QtErrorMessageBox(QMessageBox):
    def __init__(
        self,
        exc_info: Optional[ExcInfo] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setIcon(QMessageBox.Warning)
        if exc_info:
            self.format_exc_info(exc_info)

    def format_exc_info(self, exc_info: ExcInfo):
        etype, value, tb = exc_info
        title = camel_to_spaces(etype.__name__ if etype else 'Error')
        self.setWindowTitle(title or "Napari Error")
        self.setText(str(value))

        info = getattr(value, 'info', None)
        if info:
            self.setInformativeText(info + "\n")

        detail = "".join(traceback.format_exception(*exc_info))
        if detail:
            self.setDetailedText(detail)
            # FIXME
            self.setStyleSheet(
                """QTextEdit{
                    min-width: 800px;
                    font-size: 12px;
                    font-weight: 400;
                }"""
            )


class ExceptionHandler(QObject):
    """General class to handle all raise exception errors in the GUI"""

    error = Signal(tuple)

    def __init__(self):
        super().__init__()
        # could add a lot more logic about when we actually show the dialog
        self.error.connect(self._show_error_dialog)

    def handler(
        self,
        etype: Type[BaseException],
        value: BaseException,
        tb: TracebackType,
    ):
        # etype.__module__ contains the module raising the error
        # Custom exception classes can have different behavior
        # can add custom exception handlers here ...
        self.error.emit((etype, value, tb))

    def _show_error_dialog(self, exc_info: ExcInfo):
        parent = None
        for wdg in QApplication.topLevelWidgets():
            if isinstance(wdg, QMainWindow):
                parent = wdg
                break

        QtErrorMessageBox(exc_info, parent=parent).exec_()

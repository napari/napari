import logging
import traceback
from types import TracebackType
from typing import Type, Optional, Tuple

from qtpy.QtCore import QObject, Signal

from .qt_error_notification import NapariNotification


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
        text = "".join(traceback.format_exception(etype, value, tb))
        self.error.emit((etype, value, tb))
        if session_is_interactive():
            logging.error("Unhandled exception:\n%s", text)
        else:
            self._show_error_dialog(value)

    def _show_error_dialog(self, exception: BaseException):
        self.message = NapariNotification(*parse_exception(exception))
        self.message.show()


def session_is_interactive() -> bool:
    import sys
    import builtins

    if '__IPYTHON__' in dir(builtins):
        return True
    if bool(getattr(sys, 'ps1', sys.flags.interactive)):
        return True
    return False


def parse_exception(
    exception: BaseException,
) -> Tuple[str, str, Optional[str], Tuple]:
    msg = getattr(exception, 'message', str(exception))
    severity = getattr(exception, 'severity', 'WARNING')
    source = None
    actions = getattr(exception, 'actions', ())
    return msg, severity, source, actions

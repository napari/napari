import logging
import os
import sys
import traceback
from types import TracebackType
from typing import Optional, Type

from qtpy.QtCore import QObject, Signal

from .dialogs.qt_error_notification import NapariNotification


def _set_true(var) -> bool:
    return os.getenv(var) in ('1', 'True')


def _set_false(var) -> bool:
    return os.getenv(var) in ('0', 'False')


class ExceptionHandler(QObject):
    """General class to handle all uncaught exceptions in the Qt event loop.

    Parameters
    ----------
    parent : QObject, optional
        parent object, by default None
    gui_exceptions : bool, optional
        Whether to show exceptions as, by default True.  May be overriden by
        environment variable: ``NAPARI_CATCH_ERRORS=1`
        Note: this ``False`` by default in ``gui_qt()`` (the main
        instantiator of this class), but it is ``True`` in ``napari.__main__``.
        As a result, exceptions will be shown in the GUI only (mostly) when
        running napari as ``napari`` or ``python -m napari`` from the command
        line.

    Attributes
    ----------
    exit_on_error : bool
        If True exit the program after reporting an exception. It will be
        set True if the environment variable NAPARI_EXIT_ON_ERROR is set
        to "1" or "True".
    """

    error = Signal(tuple)
    message: Optional[NapariNotification] = None

    def __init__(self, parent=None, *, gui_exceptions=True):
        super().__init__(parent)
        self.gui_exceptions: bool = (
            False if _set_false("NAPARI_CATCH_ERRORS") else gui_exceptions
        )
        self.exit_on_error: bool = _set_true("NAPARI_EXIT_ON_ERROR")

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
        if isinstance(value, KeyboardInterrupt):
            print("Closed by KeyboardInterrupt", file=sys.stderr)
            sys.exit(1)
        if self.gui_exceptions:
            self._show_error_dialog(value)
        else:
            text = "".join(traceback.format_exception(etype, value, tb))
            logging.error("Unhandled exception:\n%s", text)
        self.error.emit((etype, value, tb))

        if self.exit_on_error:
            sys.exit(1)

    def _show_error_dialog(self, exception: BaseException):
        self.message = NapariNotification.from_exception(exception)
        self.message.show()

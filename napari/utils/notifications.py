from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from enum import auto
from types import TracebackType
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

from .events import Event, EventEmitter
from .misc import StringEnum

name2num = {
    'error': 40,
    'warning': 30,
    'info': 20,
    'debug': 10,
    'none': 0,
}


class NotificationSeverity(StringEnum):
    """Severity levels for the notification dialog.  Along with icons for each."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    DEBUG = auto()
    NONE = auto()

    def as_icon(self):
        return {
            self.ERROR: "ⓧ",
            self.WARNING: "⚠️",
            self.INFO: "ⓘ",
            self.DEBUG: "🐛",
            self.NONE: "",
        }[self]

    def __lt__(self, other):
        return name2num[str(self)] < name2num[str(other)]

    def __le__(self, other):
        return name2num[str(self)] <= name2num[str(other)]

    def __gt__(self, other):
        return name2num[str(self)] > name2num[str(other)]

    def __ge__(self, other):
        return name2num[str(self)] >= name2num[str(other)]


ActionSequence = Sequence[Tuple[str, Callable[[], None]]]


class Notification(Event):
    """A Notifcation event.  Usually created by :class:`NotificationManager`.

    Parameters
    ----------
    message : str
        The main message/payload of the notification.
    severity : str or NotificationSeverity, optional
        The severity of the notification, by default
        `NotificationSeverity.WARNING`.
    actions : sequence of tuple, optional
        Where each tuple is a `(str, callable)` 2-tuple where the first item
        is a name for the action (which may, for example, be put on a button),
        and the callable is a callback to perform when the action is triggered.
        (for example, one might show a traceback dialog). by default ()
    """

    def __init__(
        self,
        message: str,
        severity: Union[
            str, NotificationSeverity
        ] = NotificationSeverity.WARNING,
        actions: ActionSequence = (),
        **kwargs,
    ):
        self.severity = NotificationSeverity(severity)
        super().__init__(type=str(self.severity).lower(), **kwargs)
        self.message = message
        self.actions = actions

        # let's store when the object was created;
        self.date = datetime.now()

    @classmethod
    def from_exception(cls, exc: BaseException, **kwargs) -> 'Notification':
        return ErrorNotification(exc, **kwargs)

    @classmethod
    def from_warning(cls, warning: Warning, **kwargs) -> 'Notification':
        return WarningNotification(warning, **kwargs)


class ErrorNotification(Notification):
    exception: BaseException

    def __init__(self, exception: BaseException, *args, **kwargs):
        msg = getattr(exception, 'message', str(exception))
        actions = getattr(exception, 'actions', ())
        super().__init__(msg, NotificationSeverity.ERROR, actions)
        self.exception = exception

    def as_html(self):
        from ._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=True)

    def __str__(self):
        from ._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=False)


class WarningNotification(Notification):
    warning: Warning

    def __init__(
        self, warning: Warning, filename=None, lineno=None, *args, **kwargs
    ):
        msg = getattr(warning, 'message', str(warning))
        actions = getattr(warning, 'actions', ())
        super().__init__(msg, NotificationSeverity.WARNING, actions)
        self.warning = warning
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        category = type(self.warning).__name__
        return f'{self.filename}:{self.lineno}: {category}: {self.warning}!'


class NotificationManager:
    """
    A notification manager, to route all notifications through.

    Only one instance is in general available through napari; as we need
    notification to all flow to a single location that is registered with the
    sys.except_hook  and showwarning hook.

    This can and should be used a context manager; the context manager will
    properly re-entered, and install/remove hooks and keep them in a stack to
    restore them.

    While it might seem unnecessary to make it re-entrant; or to make the
    re-entrancy no-op; one need to consider that this could be used inside
    another context manager that modify except_hook and showwarning.

    Currently the original except and show warnings hooks are not called; but
    this could be changed in the future; this poses some questions with the
    re-entrency of the hooks themselves.
    """

    records: List[Notification]
    _instance: Optional[NotificationManager] = None

    def __init__(self) -> None:
        self.records: List[Notification] = []
        self.exit_on_error = os.getenv('NAPARI_EXIT_ON_ERROR') in ('1', 'True')
        self.notification_ready = self.changed = EventEmitter(
            source=self, event_class=Notification
        )
        self._originals_except_hooks = []
        self._original_showwarnings_hooks = []

    def __enter__(self):
        self.install_hooks()
        return self

    def __exit__(self, *args, **kwargs):
        self.restore_hooks()

    def install_hooks(self):
        """
        Install a sys.excepthook and a showwarning  hook to display any message
        in the UI, storing the previous hooks to be restored if necessary
        """

        self._originals_except_hooks.append(sys.excepthook)
        sys.excepthook = self.receive_error

        self._original_showwarnings_hooks.append(warnings.showwarning)
        warnings.showwarning = self.receive_warning

    def restore_hooks(self):
        """
        Remove hooks installed by `install_hooks` and restore previous hooks.
        """
        sys.excepthook = self._originals_except_hooks.pop()

        warnings.showwarning = self._original_showwarnings_hooks.pop()

    def dispatch(self, notification: Notification):
        self.records.append(notification)
        self.notification_ready(notification)

    def receive_error(
        self,
        exctype: Type[BaseException],
        value: BaseException,
        traceback: TracebackType,
    ):
        if isinstance(value, KeyboardInterrupt):
            sys.exit("Closed by KeyboardInterrupt")
        if self.exit_on_error:
            sys.__excepthook__(exctype, value, traceback)
            sys.exit("Exit on error")
        try:
            self.dispatch(Notification.from_exception(value))
        except Exception:
            pass

    def receive_warning(
        self,
        message: Warning,
        category: Type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line=None,
    ):
        self.dispatch(
            Notification.from_warning(
                message, filename=filename, lineno=lineno
            )
        )

    def receive_info(self, message: str):
        self.dispatch(Notification(message, 'INFO'))


notification_manager = NotificationManager()


def show_info(message: str):
    notification_manager.receive_info(message)


def show_console_notification(notification: Notification):
    from .settings import SETTINGS

    if notification.severity < SETTINGS.application.console_notification_level:
        return

    print(notification)

import os
import sys
import warnings
from enum import auto
from time import time
from types import TracebackType
from typing import Any, Callable, List, Sequence, Tuple, Type, Union

from .utils.events import Event, EventEmitter
from .utils.misc import StringEnum


class NotificationSeverity(StringEnum):
    """Severity levels for the notification dialog.  Along with icons for each."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    NONE = auto()

    def as_icon(self):
        return {
            self.ERROR: "ⓧ",
            self.WARNING: "⚠️",
            self.INFO: "ⓘ",
            self.NONE: "",
        }[self]


ActionSequence = Sequence[Tuple[str, Callable[[], None]]]


# Event subclasses specific to the Canvas
class Notification(Event):
    def __init__(
        self,
        message: str,
        severity: Union[
            str, NotificationSeverity
        ] = NotificationSeverity.WARNING,
        # source: Optional[str] = None, # TODO
        actions: ActionSequence = (),
        type: str = 'notification',
        native: Any = None,
        **kwargs,
    ):

        super().__init__(type, **kwargs)
        self._time = time()
        self.message = message
        self.severity = NotificationSeverity(severity)
        # self.source = source  # TODO
        self.actions = actions

    @property
    def time(self):
        return self._time

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
        severity = getattr(exception, 'severity', 'ERROR')
        actions = getattr(exception, 'actions', ())
        super().__init__(msg, severity, actions, type='error')
        self.exception = exception


class WarningNotification(Notification):
    warning: Warning

    def __init__(self, warning: Warning, *args, **kwargs):
        msg = getattr(warning, 'message', str(warning))
        severity = getattr(warning, 'severity', 'WARNING')
        actions = getattr(warning, 'actions', ())
        super().__init__(msg, severity, actions, type='warning')
        self.warning = warning


class NotificationManager:
    record: List[Notification]
    _instance: 'NotificationManager' = None

    def __new__(cls):
        # singleton
        if cls._instance is None:
            cls._instance = object.__new__(cls)
            sys.excepthook = cls._instance.receive_error
            warnings.__showwarning__ = warnings.showwarning
            warnings.showwarning = cls._instance.receive_warning
        return cls._instance

    def __init__(self) -> None:
        self.record: List[Notification] = []
        self.exit_on_error = os.getenv('NAPARI_EXIT_ON_ERROR') in ('1', 'True')
        self.notification_ready = self.changed = EventEmitter(
            source=self, event_class=Notification
        )

    def dispatch(self, notification: Notification):
        self.record.append(notification)
        self.notification_ready(notification)

    def receive_error(
        self,
        exctype: Type[Exception] = None,
        value: Exception = None,
        traceback: TracebackType = None,
    ):
        if isinstance(value, KeyboardInterrupt):
            print("Closed by KeyboardInterrupt", file=sys.stderr)
            sys.exit(1)
        if self.exit_on_error:
            sys.__excepthook__(exctype, value, traceback)
            sys.exit(1)
        self.dispatch(Notification.from_exception(value))

    def receive_warning(
        self,
        message: Warning,
        category: Type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line=None,
    ):
        self.dispatch(Notification.from_warning(message))

    def receive_info(self, message: str):
        self.dispatch(Notification(message, 'INFO'))


notification_manager = NotificationManager()


def show_info(message: str):
    notification_manager.receive_info(message)

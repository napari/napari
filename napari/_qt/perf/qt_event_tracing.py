"""A special QApplication for perfmon that traces events.

This file defines QApplicationWithTracing and convert_app_for_tracing(), both of
which we use when perfmon is enabled to time Qt Events.

When using perfmon there is a debug menu "Start Tracing" command as well as a
dockable QtPerformance widget.
"""
import sys

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

from ...utils import perf
from ...utils.translations import trans
from ..utils import delete_qapp


def convert_app_for_tracing(app: QApplication) -> QApplication:
    """If necessary replace existing app with our special tracing one.

    Parameters
    ----------
    app : QApplication
        The existing application if any.
    """
    if isinstance(app, QApplicationWithTracing):
        # We're already using QApplicationWithTracing so there is nothing
        # to do. This happens when napari is launched from the command
        # line because we create a QApplicationWithTracing in get_app.
        return app

    if app is not None:

        # We can't monkey patch QApplication.notify, since it's a SIP
        # wrapped C++ method. So we delete the current app and create a new
        # one. This must be done very early before any Qt objects are
        # created or we will crash!
        delete_qapp(app)

    return QApplicationWithTracing(sys.argv)


class QApplicationWithTracing(QApplication):
    """Extend QApplication to trace Qt Events.

    This QApplication wraps a perf_timer around the normal notify().

    Notes
    -----
    Qt Event handling is nested. A call to notify() can trigger other calls to
    notify() prior to the first one finishing, even several levels deep.

    The hierarchy of timers is displayed correctly in the chrome://tracing GUI.
    Seeing the structure of the event handling hierarchy can be very informative
    even apart from the actual timing numbers, which is why we call it "tracing"
    instead of just "timing".
    """

    def notify(self, receiver, event):
        """Trace events while we handle them."""
        timer_name = _get_event_label(receiver, event)

        # Time the event while we handle it.
        with perf.perf_timer(timer_name, "qt_event"):
            return QApplication.notify(self, receiver, event)


class EventTypes:
    """Convert event type to a string name.

    Create event type to string mapping once on startup. We want human-readable
    event names for our timers. PySide2 does this for you but PyQt5 does not:

    # PySide2
    str(QEvent.KeyPress) -> 'PySide2.QtCore.QEvent.Type.KeyPress'

    # PyQt5
    str(QEvent.KeyPress) -> '6'

    We use this class for PyQt5 and PySide2 to be consistent.
    """

    def __init__(self):
        """Create mapping for all known event types."""
        self.string_name = {}
        for name in vars(QEvent):
            attribute = getattr(QEvent, name)
            if type(attribute) == QEvent.Type:
                self.string_name[attribute] = name

    def as_string(self, event: QEvent.Type) -> str:
        """Return the string name for this event.

        event : QEvent.Type
            Return string for this event type.
        """
        try:
            return self.string_name[event]
        except KeyError:
            return trans._("UnknownEvent:{event}", event=event)


EVENT_TYPES = EventTypes()


def _get_event_label(receiver: QWidget, event: QEvent) -> str:
    """Return a label for this event.

    Parameters
    ----------
    receiver : QWidget
        The receiver of the event.
    event : QEvent
        The event name.

    Returns
    -------
    str
        Label to display for the event.

    Notes
    -----
    If no object we return <event_name>.
    If there's an object we return <event_name>:<object_name>.

    Combining the two names with a colon is our own made-up format. The name
    will show up in chrome://tracing and our QtPerformance widget.
    """
    event_str = EVENT_TYPES.as_string(event.type())

    try:
        # There may or may not be a receiver object name.
        object_name = receiver.objectName()
    except AttributeError:
        # Ignore "missing objectName attribute" during shutdown.
        object_name = None

    if object_name:
        return f"{event_str}:{object_name}"

    # There was no object (pretty common).
    return event_str

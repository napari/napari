"""A special QApplication for perfmon that traces events.

This file defines QApplicationWithTracing which we use when perfmon is
enabled to time Qt Events.

When using perfmon there is a debug menu "Start Tracing" command as well as a
dockable QtPerformance widget.
"""
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

from napari.utils import perf
from napari.utils.translations import trans


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

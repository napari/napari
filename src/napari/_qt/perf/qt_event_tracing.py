"""A special QApplication for perfmon that traces events.

This file defines QApplicationWithTracing which we use when perfmon is
enabled to time Qt Events.

When using perfmon there is a debug menu "Start Tracing" command as well as a
dockable QtPerformance widget.
"""

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

from napari.utils import perf


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
        with perf.perf_timer(timer_name, 'qt_event'):
            return QApplication.notify(self, receiver, event)


def _event_type_name(event_type: QEvent.Type) -> str:
    """Return a human-readable name for this event type.

    Qt6 bindings expose event types as Python enums. Event types that are
    not members of QEvent.Type (for example, registered custom events) have
    no meaningful name.
    """
    name = getattr(event_type, 'name', None)
    if name and not name.isdigit():
        return name
    return f'UnknownEvent:{event_type}'


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
    event_str = _event_type_name(event.type())

    try:
        # There may or may not be a receiver object name.
        object_name = receiver.objectName()
    except AttributeError:
        # Ignore "missing objectName attribute" during shutdown.
        object_name = None

    if not object_name:
        # use class for object without set name.
        try:
            object_name = str(receiver.__class__.__name__)
        except AttributeError:
            # do not crash in using some strange class.
            object_name = None

    if object_name:
        return f'{event_str}:{object_name}'

    # There was no object (pretty common).
    return event_str

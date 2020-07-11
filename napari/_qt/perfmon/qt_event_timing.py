"""A special QApplication for perfmon.

This file defines QApplicationWithTiming and convert_app_for_timing(), both of
which we use when perfmon is enabled to time Qt Events.

Perf timers power the debug menu's "Start Tracing" feature as well as the
dockable QtPerformance widget.
"""
import sys

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QWidget

from ..utils import perf


def convert_app_for_timing(app: QApplication) -> QApplication:
    """If necessary replace existing app with our special perfmon one.

    Parameters
    ----------
    app : QApplication
        The existing application if any.
    """
    if isinstance(app, QApplicationWithTiming):
        # We're already using QApplicationWithTiming so there is nothing
        # to do. This happens when napari is launched from the command
        # line because we create a QApplicationWithTiming in gui_qt.
        return app

    if app is not None:

        # Because we can't monkey patch QApplication.notify, since it's a
        # SIP wrapped C++ method, we delete the current app and create a new one.
        # This must be done very early before any Qt objects are created.
        import sip

        sip.delete(app)

    # Is it right to pass in sys.argv here? I think so if there are any
    # Qt flags on there?
    return QApplicationWithTiming(sys.argv)


class QApplicationWithTiming(QApplication):
    """Extend QApplication to time Qt Events.

    This QApplication times how long the normal notify() method takes.

    Notes
    -----
    Qt Event handling is nested. A call to notify() can trigger other calls to
    notify() prior to the first one finishing, even several levels deep.

    The hierarchy of timers is displayed correctly in the chrome://tracing GUI.
    Seeing the structure of the event handling hierarchy can be very informative
    even apart from the actual timing numbers.
    """

    def notify(self, receiver, event):
        """Time events while we handle them."""
        timer_name = _get_timer_name(receiver, event)

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
            return f"UnknownEvent:{event}"


EVENT_TYPES = EventTypes()


def _get_timer_name(receiver: QWidget, event: QEvent) -> str:
    """Return a name for this event.

    Parameters
    ----------
    receiver : QWidget
        The receiver of the event.
    event : QEvent
        The event name.

    Returns
    -------
    str
        The timer's name

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

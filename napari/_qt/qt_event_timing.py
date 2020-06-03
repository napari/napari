"""A special QApplication for perfmon.

Defines QApplicationWithTiming and convert_app_for_timing(), both of which we use
when perfmon is enabled to time Qt Events.

Perf timers power the debug menu's "Start Tracing" feature as well as the
dockable QtPerformance widget.
"""
from qtpy.QtWidgets import QApplication

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

        # Because we can't monkey patch QApplication.notify (since it's a
        # wrapped C++ method?) we delete the current app and create a new one.
        # This must be done very early before any Qt objects are created.
        import sip

        sip.delete(app)

    return QApplicationWithTiming([])


class QApplicationWithTiming(QApplication):
    """Extend QApplication to time Qt Events.

    Our QApplication times how long the normal notify() method takes.

    Notes
    -----
    Qt Event handling is nested. A call to notify() can trigger other calls to
    notify() prior to the first one finishing, even several levels deep.

    The hierarchy of timers is displayed correctly in the chrome://tracing GUI.
    Seeing the structure of the event handling hierarchy can be very informative
    even apart from the timing numbers.
    """

    def notify(self, receiver, event):
        """Time events while we handle them."""
        timer_name = _get_timer_name(receiver, event)

        # Time the event while we handle it.
        with perf.perf_timer(timer_name, "qt_event"):
            return QApplication.notify(self, receiver, event)


def _get_timer_name(receiver, event) -> str:
    """Return a name for this event.

    Parameters
    ----------
    receiver : QWidget
        The receiver of the event.
    event : QEvent
        The event name.

    Returns
    -------
    timer_name : str

    Notes
    -----
    If no object we return <event_name>.
    If there's an object we return <event_name>:<object_name>.

    This our own made up format we can revise as needed.
    """
    # For an event.type() like "PySide2.QtCore.QEvent.Type.WindowIconChange"
    # we set event_str to just the final "WindowIconChange" part.
    event_str = str(event.type()).split(".")[-1]

    try:
        # There may or may not be a receiver object name.
        object_name = receiver.objectName()
    except AttributeError:
        # During shutdown the call to receiver.objectName() can fail with
        # "missing objectName attribute". Ignore and assume no object name.
        object_name = None

    if object_name:
        return f"{event_str}:{object_name}"

    # Many events have no object, only an event string.
    return event_str

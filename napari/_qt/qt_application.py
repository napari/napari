"""Defines our QtApplication.

In the normal case QtApplication is just QApplication.

However if the NAPARI_PERFMON environment variable is set then QtApplication is
the PerfMonApplication defined below, which adds timing of every QtEvent.

This is a WIP as we experiment with performance monitoring.
"""
import os
import time

from qtpy.QtWidgets import QApplication

from ..utils.perf_timers import TIMERS


def _get_event_name(event, receiver):
    """Get single compound name for this event.

    If there is no object we return just <event_name>.
    If there is an object we do <event_name>:<object_name>

    This is a made up format to have just a single string, but if we end up
    parsing this string, we might want to just keep it as a tuple.
    """
    # For an event.type() like "PySide2.QtCore.QEvent.Type.WindowIconChange"
    # We just use the final "WindowIconChange" part.
    event_str = str(event.type()).split(".")[-1]

    # Try since we sometimes get:
    # AttributeError: 'PySide2.QtGui.QMouseEvent' object has no attribute 'objectName'
    # Is that because the receiver is an event in that case? Not sure yet.
    try:
        object_name = receiver.objectName()
    except AttributeError:
        print("ATTRIBUTE ERROR type = ", type(receiver))
        object_name = None

    # Make up a colon separated syntax, although we could keep these
    # separate if that made things down the line easier.
    if object_name:
        return f"{event_str}:{object_name}"

    # Frequently events have no object.
    return event_str


class PerfmonApplication(QApplication):
    """Extend QApplication to time events.

    Performance monitoring is a WIP. The goal is provide insight into what is
    causing Napari to run slowly or not smoothly.

    There are 3 main parts to performance monitoring today:
    1) PerfMonApplication: times events, records them with PerfTimers.
    2) PerfTimers: stores timing data, optionally writes to chrome://tracing.
    3) QtPerformance: dockable widget which displays some PerfTime data.

    In the future we will have PerfTimers which are not Qt Events. Where we
    explicitly time a block of code or an IO event.

    Nesting: Note that Qt Event handling is nested. A call to notify() can
    trigger other calls to notify() prior to the first call finishing. This
    hierarchy of event processing is visible in chrome://tracing.

    Some TBD items:
    1) Measure the overhead of monitoring. Hopefully some monitoring can be
    left enabled at all times, but it might be more minimal that what we
    are doing today guarded by a environment variables.

    2) At other times of PerfTimers that are not Qt Events, for example
    timing a block of code or an IO operation.

    3) Handling threading, chrome://tracing can supports threads.
    """

    def notify(self, receiver, event):
        """Time every event."""
        # Must grab these before calling notify().
        name = _get_event_name(event, receiver)

        # Notify as usual while timing it.
        start_ns = time.perf_counter_ns()
        ret_value = QApplication.notify(self, receiver, event)
        end_ns = time.perf_counter_ns()

        # For now record every event. Later if not doing a full trace we
        # will probably only record times over some threshold duration.
        TIMERS.record(name, start_ns, end_ns)
        return ret_value


USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

if USE_PERFMON:
    # Use our performance monitoring version.
    print("Performance Monitoring: ENABLED")
    QtApplication = PerfmonApplication
else:
    # Use the normal stock QApplication
    QtApplication = QApplication

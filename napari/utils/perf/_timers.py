"""PerfTimers class and global instance.
"""
import os

from ._compat import perf_counter_ns
from ._event import InstantEvent, PerfEvent
from ._stat import Stat
from ._trace_file import PerfTraceFile

USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"


class PerfTimers:
    """Timers for performance monitoring.

    For each PerfEvent recorded we do two things:
    1) Update our self.timers dictionary (always).
    2) Write to a trace file (optional if recording one).

    Anyone can record a timing event, but these are 3 common ways:
    1) Our custom QtApplication times Qt Events.
    2) Our perf_timer context object times blocks of code.
    3) Our perf_func decorator can time functions.

    The QtPerformance Widget goes through our self.timers looking for long events
    and prints them to a log window. Then it clears the timers.

    Attributes
    ----------
    timers : dict
        Maps a timer name to a SimpleStat object.
    trace_file : PerfTraceFile, optional
        The tracing file we are writing to if any.

    Notes
    -----
    Chrome deduces nesting based on the start and end times of each timer. The
    chrome://tracing GUI shows the nesting as stacks of colored rectangles.

    However our self.timers dictionary and thus our QtPerformance widget do not
    currently understand nesting. So if they say two timers each took 1ms, you
    can't tell if one called the other or not.

    Despite this limitation when the QtPerformance widget reports slow timers it
    still gives you a good idea what was slow. And then you can use the
    chrome://tracing GUI to see the full story.
    """

    def __init__(self):
        """Create PerfTimers.
        """
        # Maps a timer name to one Stat object.
        self.timers = {}

        # Menu item "Debug -> Record Trace File..." starts a trace.
        self.trace_file = None

    def add_event(self, event: PerfEvent):
        """Add one completed event.

        Parameters
        ----------
        event : PerfEvent
            Add this event.
        """
        # Add event if tracing.
        if self.trace_file is not None:
            self.trace_file.add_event(event)

        if event.phase == "X":  # Complete Event
            # Update our self.timers (in milliseconds).
            name = event.name
            duration_ms = event.duration_ms
            if name in self.timers:
                self.timers[name].add(duration_ms)
            else:
                self.timers[name] = Stat(duration_ms)

    def add_instant_event(self, name: str, **kwargs):
        """Add one instant event.

        Parameters
        ----------
        event : PerfEvent
            Add this event.
        """
        self.add_event(InstantEvent(name, perf_counter_ns(), **kwargs))

    def clear(self):
        """Clear all timers.
        """
        # After the GUI displays timing information it clears the timers
        # so that we start accumulating fresh information.
        self.timers.clear()

    def start_trace_file(self, path: str) -> None:
        """Start recording a trace file to disk.

        Parameters
        ----------
        path : str
            Write the trace to this path.
        """
        self.trace_file = PerfTraceFile(path)

    def stop_trace_file(self) -> None:
        """Stop recording a trace file.
        """
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None


if USE_PERFMON:
    # The one global instance
    timers = PerfTimers()

    def add_instant_event(name: str, **kwargs):
        timers.add_instant_event(name, **kwargs)


else:
    # No one should be access this since they are disabled.
    timers = None

    def add_instant_event(name: str, **kwargs):
        pass

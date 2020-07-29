"""PerfTimers class and global instance.
"""
import contextlib
import os
from typing import Optional

from ._compat import perf_counter_ns
from ._event import InstantEvent, PerfEvent
from ._stat import Stat
from ._trace_file import PerfTraceFile

USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"


class PerfTimers:
    """Timers for performance monitoring.

    For each added PerfEvent we do two things:
    1) Update our self.timers dictionary (always).
    2) Write to a trace file (optionally if recording one).

    You can add a PerfEvent completely by hand by creating a
    PerfEvent object and calling add_event(). However typically
    you add PerfEvents one of three more automatic ways:

    1) Enable timing of Qt Events using QApplicationWithTracing.
    2) Patch in perf_timers using the perfmon config file.
    3) Add perf_timer context objects by hand.

    Methods 1 and 2 result in zero overhead if perfmon is disabled,
    but 3 results in a tiny amount of overhead (1 usec per timer)
    therefore best practice is remove manual perf_timers before
    merging into master. Consider them like debug prints.

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

    @contextlib.contextmanager
    def perf_timer(
        name: str,
        category: Optional[str] = None,
        print_time: bool = False,
        **kwargs,
    ):
        """Time a block of code.

        Parameters
        ----------
        name : str
            The name of this timer.
        category : str
            Comma separated categories such has "render,update".
        **kwargs : dict
            Additional keyword arguments for the "args" field of the event.

        Examples
        --------
        with perf_timer("draw"):
            draw_stuff()
        """
        start_ns = perf_counter_ns()
        yield
        end_ns = perf_counter_ns()
        event = PerfEvent(name, start_ns, end_ns, **kwargs)
        timers.add_event(event)
        if print_time:
            ms = (end_ns - start_ns) / 1e6
            print(f"{name} {ms}ms")


else:
    # No one should be access this since they are disabled.
    timers = None

    def add_instant_event(name: str, **kwargs):
        pass

    # contextlib.nullcontext does not work with kwargs, so we just
    # create a do-nothing context object. This is not zero overhead
    # but it's very low, about 1 microsecond? But because it's not
    # zero it's best practice not to commit perf_timers, think of
    # them like debug prints, add while investigating a problem
    # but then remove before committing/merging.
    @contextlib.contextmanager
    def perf_timer(name: str, category: Optional[str] = None, **kwargs):
        yield

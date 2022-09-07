"""PerfTimers class and global instance.
"""
import contextlib
import os
from time import perf_counter_ns
from typing import Dict, Optional

from ._event import PerfEvent
from ._stat import Stat
from ._trace_file import PerfTraceFile

USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"


class PerfTimers:
    """Timers for performance monitoring.

    Timers are best added using the perfmon config file, which will
    monkey-patch the timers into the code at startup. See
    napari.utils.perf._config for details.

    The collecting timing information can be used in two ways:
    1) Writing a JSON trace file in Chrome's Tracing format.
    2) Napari's real-time QtPerformance widget.

    Attributes
    ----------
    timers : Dict[str, Stat]
        Statistics are kept on each timer.
    trace_file : Optional[PerfTraceFile]
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
        """Create PerfTimers."""
        # Maps a timer name to one Stat object.
        self.timers: Dict[str, Stat] = {}

        # Menu item "Debug -> Record Trace File..." starts a trace.
        self.trace_file: Optional[PerfTraceFile] = None

    def add_event(self, event: PerfEvent) -> None:
        """Add one performance event.

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

    def add_instant_event(self, name: str, **kwargs) -> None:
        """Add one instant event.

        Parameters
        ----------
        name : PerfEvent
            Add this event.
        **kwargs
            Arguments to display in the Args section of the Tracing GUI.
        """
        now = perf_counter_ns()
        self.add_event(PerfEvent(name, now, now, phase="I", **kwargs))

    def add_counter_event(self, name: str, **kwargs: Dict[str, float]) -> None:
        """Add one counter event.

        Parameters
        ----------
        name : str
            The name of this event like "draw".
        **kwargs : Dict[str, float]
            The individual counters for this event.

        Notes
        -----
        For example add_counter_event("draw", triangles=5, squares=10).
        """
        now = perf_counter_ns()
        self.add_event(PerfEvent(name, now, now, phase="C", **kwargs))

    def clear(self):
        """Clear all timers."""
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
        """Stop recording a trace file."""
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None


@contextlib.contextmanager
def block_timer(
    name: str,
    category: Optional[str] = None,
    print_time: bool = False,
    **kwargs,
):
    """Time a block of code.

    block_timer can be used when perfmon is disabled. Use perf_timer instead
    if you want your timer to do nothing when perfmon is disabled.

    Notes
    -----
    Most of the time you should use the perfmon config file to monkey-patch
    perf_timer's into methods and functions. Then you do not need to use
    block_timer or perf_timer context objects explicitly at all.

    Parameters
    ----------
    name : str
        The name of this timer.
    category : str
        Comma separated categories such has "render,update".
    print_time : bool
        Print the duration of the timer when it finishes.
    **kwargs : dict
        Additional keyword arguments for the "args" field of the event.

    Examples
    --------

    .. code-block:: python

        with block_timer("draw") as event:
            draw_stuff()
        print(f"The timer took {event.duration_ms} milliseconds.")

    """
    start_ns = perf_counter_ns()

    # Pass in start_ns for start and end, we call update_end_ns
    # once the block as finished.
    event = PerfEvent(name, start_ns, start_ns, category, **kwargs)
    yield event

    # Update with the real end time.
    event.update_end_ns(perf_counter_ns())

    if timers:
        timers.add_event(event)
    if print_time:
        print(f"{name} {event.duration_ms:.3f}ms")


def _create_timer():
    # The one global instance
    timers = PerfTimers()

    # perf_timer is enabled
    perf_timer = block_timer

    def add_instant_event(name: str, **kwargs):
        """Add one instant event.

        Parameters
        ----------
        name : PerfEvent
            Add this event.
        **kwargs
            Arguments to display in the Args section of the Chrome Tracing GUI.
        """
        timers.add_instant_event(name, **kwargs)

    def add_counter_event(name: str, **kwargs: Dict[str, float]):
        """Add one counter event.

        Parameters
        ----------
        name : str
            The name of this event like "draw".
        **kwargs : Dict[str, float]
            The individual counters for this event.

        Notes
        -----
        For example add_counter_event("draw", triangles=5, squares=10).
        """
        timers.add_counter_event(name, **kwargs)

    return timers, perf_timer, add_instant_event, add_counter_event


if USE_PERFMON:
    timers, perf_timer, add_instant_event, add_counter_event = _create_timer()

else:
    # Make sure no one accesses the timers when they are disabled.
    timers = None

    def add_instant_event(name: str, **kwargs) -> None:
        pass

    def add_counter_event(name: str, **kwargs: Dict[str, float]) -> None:
        pass

    # perf_timer is disabled. Using contextlib.nullcontext did not work.
    @contextlib.contextmanager
    def perf_timer(name: str, category: Optional[str] = None, **kwargs):
        yield

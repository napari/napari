"""PerfEvent class.
"""
import os
import threading
from collections import namedtuple
from typing import Dict, Optional

# A span of time for an event.
Span = namedtuple("Span", "start_ns end_ns")

# A origin for an event.
Origin = namedtuple("Origin", "process_id thread_id")


class PerfEvent:
    """One perf event represents a span of time.

    Parameters
    ----------
    name : str
        The name of this event like "draw".
    start_ns : int
        Start time in nanoseconds.
    end_ns : int
        End time in nanoseconds.
    process_id : int
        The process id that produced the event.
    thread_id : int
        The thread id that produced the event.
    category :str
        Comma separated categories such has "render,update".
    **kwargs : dict
        Additional keyword arguments for the "args" field of the event.


    Attributes
    ----------
    name : str
        The name of this event like "draw".
    span : Span
        The time span when the event happened.
    category :str
        Comma separated categories such has "render,update".
    phase : str
        The chrome://tracing "phase" (event type). The spec defines
        around 20 phases we only support two right now:
             "X" - Complete Events
             "I" - Instant Events
    owner : Owner
        The process and thread that produced the event.
    arg : dict
        Arbitrary keyword arguments for this event.

    Notes
    -----
    The time stamps are from perf_counter_ns() and do not indicate time of
    day. The origin is arbitrary, but subtracting two counters results in a
    valid span of wall clock time. If start is the same as end the event
    was instant.

    The full Chrome Tracing spec is a Google Doc called "Trace Event Format",
    search for that phrase to find it.
    """

    def __init__(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        category: Optional[str] = None,
        process_id: int = os.getpid(),
        thread_id: int = threading.get_ident(),
        phase: str = "X",  # "X" is a "complete event" in their spec.
        **kwargs: dict,
    ):
        self.name: str = name
        self.span: Span = Span(start_ns, end_ns)
        self.category: str = category
        self.origin: Origin = Origin(process_id, thread_id)
        self.args = kwargs
        self.phase: str = phase

    @property
    def start_us(self):
        """Start time in microseconds."""
        return self.span.start_ns / 1e3

    @property
    def start_ms(self):
        """Start time in milliseconds."""
        return self.span.start_ns / 1e6

    @property
    def duration_ns(self):
        """Duration in nanoseconds."""
        return self.span.end_ns - self.span.start_ns

    @property
    def duration_us(self):
        """Duration in microseconds."""
        return self.duration_ns / 1e3

    @property
    def duration_ms(self):
        """Duration in milliseconds."""
        return self.duration_ns / 1e6


class InstantEvent(PerfEvent):
    """An instant event draws as a vertical line in the Chrome Tracing GUI.

    Parameters
    ----------
    name : str
        The name of this event.
    time_ns : int
        The time in nanoseconds when the event happened.
    kwargs : dict
        Arguments to display in the Args section of the GUI.
    """

    def __init__(self, name: str, time_ns: int, **kwargs):
        # Instant events are phase "I" in their trace space.
        super().__init__(name, time_ns, time_ns, phase="I", **kwargs)


class CounterEvent(PerfEvent):
    """A counter event draws as a little bar graph in the Chrome Tracing GUI.

    It draws as a bar graph if there is one counter, or a stacked bar
    graph if there are multiple counters.

    Parameters
    ----------
    name : str
        The name of this event.
    time_ns : int
        The time in nanoseconds when the event happened.
    kwargs : dict
        The names and values of the counters.
    """

    def __init__(self, name: str, time_ns: int, **kwargs: Dict[str, float]):
        # Counter events are phase "C" in their trace space.
        super().__init__(name, time_ns, time_ns, phase="C", **kwargs)

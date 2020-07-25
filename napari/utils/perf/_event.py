"""PerfEvent class.
"""
import os
import threading
from typing import Optional


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
    phase : str
        The chrome://tracing "phase" (event type). The spec defines
        around 20 phases we only support two right now:
             "X" - Complete Events
             "I" - Instant Events
    args : dict
        Keyword arguments for this event, visible when you click on the event
        in the chrome://tracing GUI.

    Notes
    -----
    The time stamps are from perf_counter_ns() and do not indicate time of
    day. The origin is arbitrary, but subtracting two counters results in
    a span of wall clock time.

    Google for "Trace Event Format" for the full chrome://tracing spec.
    """

    def __init__(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        category: Optional[str] = None,
        process_id: int = os.getpid(),
        thread_id: int = threading.get_ident(),
        **kwargs: dict,
    ):
        self.name = name
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.category = category
        self.args = kwargs
        self.process_id = process_id
        self.thread_id = thread_id
        self.phase = "X"  # Complete Event

    @property
    def start_us(self):
        return self.start_ns / 1e3

    @property
    def start_ms(self):
        return self.start_ns / 1e6

    @property
    def duration_ns(self):
        return self.end_ns - self.start_ns

    @property
    def duration_us(self):
        return self.duration_ns / 1e3

    @property
    def duration_ms(self):
        return self.duration_ns / 1e6


class InstantEvent(PerfEvent):
    def __init__(self, name: str, time_ns: int, **kwargs):
        super().__init__(name, time_ns, time_ns, **kwargs)
        self.phase = "I"  # instant event

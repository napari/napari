"""PerfTraceFile class to write JSON files in the chrome://tracing format.
"""
import json
import os
import threading

from ._compat import perf_counter_ns
from ._event import PerfEvent


class PerfTraceFile:
    """Writes a chrome://tracing formatted JSON file.

    Chrome has a nice built-in performance tool called chrome://tracing. Chrome
    can record traces of web applications. But the format is well-documented and
    anyone can create the files and use the nice GUI. And other programs accept
    the format including:
    1) https://www.speedscope.app/ which does flamegraphs (Chrome doesn't).
    2) Qt Creator's performance tools.

    Parameters
    ----------
    path : str
        Write the trace file to this path.

    Attributes
    ----------
    zero_ns : int
        perf_counter_ns() time when we started the trace.
    pid : int
        Process ID.
    outf : file handle
        JSON file we are writing to.

    Notes
    -----
    There are two chrome://tracing formats:
    1) JSON Array Format
    2) JSON Object Format

    We are using the JSON Array Format right now, the file can be cut off at
    anytime. The other format allows for more options if we need them but must
    be closed properly.

    See the "trace_event format" Google Doc for details:
    https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md
    """

    def __init__(self, path: str):
        """Open the tracing file on disk.
        """
        # So the events we write start at t=0.
        self.zero_ns = perf_counter_ns()

        # Assume all events are from the same process for now.
        self.pid = os.getpid()

        # Start writing the file with an open bracket, per JSON Array format.
        self.outf = open(path, "w")
        self.outf.write("[\n")
        self.outf.flush()

        # Accumulate events in a list and only write at the end so the cost
        # of writing to a file does not bloat our timings.
        self.events = []

    def add_event(self, event: PerfEvent) -> None:
        """Add one perf event to the file.

        Parameters
        ----------
        event : PerfEvent
            Event to add
        """
        self.events.append(event)

    def close(self):
        """Write all stored events.
        """
        for event in self.events:
            self.write_event(event)
        self.outf.flush()

    def write_event(self, event: PerfEvent) -> None:
        """Write one perf event.

        Parameters
        ----------
        event : PerfEvent
            Event to write.
        """
        category = "none" if event.category is None else event.category

        # Event type "X" denotes a completed event. Meaning we already
        # know the duration. The format wants times in micro-seconds.
        data = {
            "pid": self.pid,
            "tid": threading.get_ident(),
            "name": event.name,
            "cat": category,
            "ph": "X",
            "ts": event.start_us,
            "dur": event.duration_us,
            "args": event.args,
        }
        json_str = json.dumps(data)

        # Write comma separated JSON objects. Note jsonlines is really a better
        # way to write JSON that can be cut off, but chrome://tracing probably
        # predates that convention.
        self.outf.write(f"{json_str},\n")

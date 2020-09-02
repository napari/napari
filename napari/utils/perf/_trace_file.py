"""PerfTraceFile class to write the chrome://tracing file format (JSON)
"""
import json
from typing import List

from ._compat import perf_counter_ns
from ._event import PerfEvent


class PerfTraceFile:
    """Writes a chrome://tracing formatted JSON file.

    Stores PerfEvents in memory, writes the JSON file in PerfTraceFile.close().

    Parameters
    ----------
    output_path : str
        Write the trace file to this path.

    Attributes
    ----------
    output_path : str
        Write the trace file to this path.
    zero_ns : int
        perf_counter_ns() time when we started the trace.
    events : List[PerfEvent]
        Process ID.
    outf : file handle
        JSON file we are writing to.

    Notes
    -----
    See the "trace_event format" Google Doc for details:
    https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md
    """

    def __init__(self, output_path: str):
        """Store events in memory and write to the file when done."""
        self.output_path = output_path

        # So the events we write start at t=0.
        self.zero_ns = perf_counter_ns()

        # Accumulate events in a list and only write at the end so the cost
        # of writing to a file does not bloat our timings.
        self.events: List[PerfEvent] = []

    def add_event(self, event: PerfEvent) -> None:
        """Add one perf event to our in-memory list.

        Parameters
        ----------
        event : PerfEvent
            Event to add
        """
        self.events.append(event)

    def close(self):
        """Close the trace file, write all events to disk."""
        event_data = [self._get_event_data(x) for x in self.events]
        with open(self.output_path, "w") as outf:
            json.dump(event_data, outf)

    def _get_event_data(self, event: PerfEvent) -> dict:
        """Return the data for one perf event.

        Parameters
        ----------
        event : PerfEvent
            Event to write.

        Returns
        -------
        dict
            The data to be written to JSON.
        """
        category = "none" if event.category is None else event.category

        data = {
            "pid": event.origin.process_id,
            "tid": event.origin.thread_id,
            "name": event.name,
            "cat": category,
            "ph": event.phase,
            "ts": event.start_us,
            "args": event.args,
        }

        # The three phase types we support.
        assert event.phase in ["X", "I", "C"]

        if event.phase == "X":
            # "X" is a Complete Event, it has a duration.
            data["dur"] = event.duration_us
        elif event.phase == "I":
            # "I is an Instant Event, it has a "scope" one of:
            #     "g" - global
            #     "p" - process
            #     "t" - thread
            # We hard code "process" right now because that's all we've needed.
            data["s"] = "p"

        return data

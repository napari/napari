"""Timers for monitoring performance.
"""
import os
import time

from .tracing import ChromeTracingFile


class PerfTimer:
    """One performance timer.

    Each PerfTimer stores the min/max/average for one value.

    We want the min/max/average because the QtPerformance UI wants to
    display those state and not just display the previous frame's
    timing information.

    So we can run QtPerformance at some lower frequence like 1Hz even
    while PerfTimers is getting populated every frame.
    """

    def __init__(self, value):
        self.min = value
        self.max = value
        self.sum = value
        self.count = 1

    def add(self, value):
        self.sum += value
        self.count += 1
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        return 0


class PerfTimers:
    """Performance Timers.

    Performance Timers are meant to be timers from various sources, Qt Events,
    blocks of code that we explicitly time, IO operations.

    This is a WIP and for now we only time Qt Events.

    Environment Variables:

    NAPARI_PERFMON_TRACE_PATH
        If set we write to this path in chrome://tracing format.
    """

    def __init__(self):
        """Create PerfTimers, start log file if we are logging.
        """
        # Key is (event_name, object_name).
        self.timers = {}
        self.trace_file = self._create_trace_file()

        # So we can start the times at zero.
        self.zero_ns = time.perf_counter_ns()

    def _create_trace_file(self) -> ChromeTracingFile:
        """Return ChromeTracingFile or None."
        """
        path = os.getenv("NAPARI_PERFMON_TRACE_PATH")
        if not path:
            return None
        return ChromeTracingFile(path)

    def record(self, name, start_ns, end_ns):
        """Record the span of one timer.
        """
        # Make the times zero based.
        start_ns = start_ns - self.zero_ns
        end_ns = end_ns - self.zero_ns
        duration_ns = end_ns - start_ns

        # Chrome tracing wants micro-seconds.
        start_us = start_ns / 1000
        duration_us = duration_ns / 1000
        if self.trace_file:
            self.trace_file.write_event(name, start_us, duration_us)

        # Our own timers are using milliseconds, for now.
        duration_ms = duration_ns / 1e6
        if name in self.timers:
            self.timers[name].add(duration_ms)
        else:
            self.timers[name] = PerfTimer(duration_ms)

    def clear(self):
        """Clear all timers.

        Once the QtPerformance widget shows the timers we clear them.
        """
        self.timers.clear()


# Only one instance today, but we could have more maybe?
TIMERS = PerfTimers()

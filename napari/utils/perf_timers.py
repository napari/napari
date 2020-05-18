"""Timers for monitoring performance.
"""
import os
import time

from .tracing import ChromeTracingFile


class PerfTimer:
    """One performance timer.

    Each PerfTimer stores the min/max/average for one timer.

    We want to track the min/max/average because the UI might be updating at a
    much slower rate than the timers. And displaying the min/max/average gives a
    better picture of what's happening than just show the duration for the
    last time it ran.

    For example we might be drawing at 60Hz but the QtPerformance widget
    is only updating at 1Hz.
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

    Performance Timers are for recording the duration of:
    1) Qt Event handling (today)
    2) Key blocks of code (future);
    3) IO operations (future)

    Environment Variables
    ---------------------

    NAPARI_PERFMON_TRACE_PATH

    Write all timers to this path in chrome://tracing format.

    Nesting:
    --------
    Chrome tracing correctly figures out nesting based on the start/end
    times of each timer. In the chrome://tracing viewer you can see the
    nesting exactly as it happened.

    Our own self.timers dictionary does not understand nesting yet.

    If two timers took 1ms but they overlapped with different names:
    <------RequestUpdate------>
    <------------Paint-------->

    Then we'll see that 2 timers that each took 1ms, even though it was the
    same 1ms. If both timers have the same name:

    <----------Resize--------->
    <----------Resize--------->

    Then we'll show the Resize event was called twice for 1ms each.

    Even though this sounds broken, for the purpose of just identifying long
    running events it still works well. But full support for nesting is something
    we could add.
    """

    def __init__(self):
        """Create PerfTimers, optionally start a trace file.
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
        duration_ns = end_ns - start_ns

        # Make the times zero based.
        start_ns = start_ns - self.zero_ns
        end_ns = end_ns - self.zero_ns

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


# Only one instance today.
TIMERS = PerfTimers()

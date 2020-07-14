"""Performance Monitoring init.

USE_PERFMON is true only if NAPARI_PERFMON environment variable is set and not
zero.

timers is an instance of PerfTimers with these methods:
    add_event(event: PerfEvent)
    clear()
    start_trace_file(path: str)
    stop_trace_file()

Use perf_timer to time blocks of code.
Use perf_func to time functions.
"""
import os

from ._compat import perf_counter_ns
from ._config import perf_config
from ._event import PerfEvent
from ._timers import timers, add_instant_event
from ._utils import perf_timer, perf_func


# If not using perfmon timers will be 100% disable with hopefully zero
# run-time impact.
USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

"""Performance Monitoring init.

USE_PERFMON is true only if NAPARI_PERFMON environment variable is set and not
zero.

timers is an instance of PerfTimers with these methods:
    add_event(event: PerfEvent)
    clear()
    start_trace_file(path: str)
    stop_trace_file()

Use perf_timer to time blocks of code.

"""
import os

from ._compat import perf_counter_ns
from ._config import perf_config
from ._event import PerfEvent
from ._timers import timers, add_instant_event

# perf_timers can be manually added to the code while debugging or
# investigating a performance issue but they should be removed prior to
# committing the code. Consider them like debug prints.
#
# An alternative to manual perf_timers is to refactor the code so the block
# of code you want to time is in its own function. Then you can patch in
# the timer using the config file.
from ._utils import perf_timer


# If not using perfmon timers will be 100% disabled with hopefully zero
# run-time impact.
USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

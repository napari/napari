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

# timers
#     The global PerfTimers instance.
#
# perf_timer
#     Context object to time a line or block of code.
#
# add_counter_event
#     Counter events appear as a little (stacked) bar graph.
#
# add_instant_event
#     Instant events appear as a vertical line.
#
# The best way to add perf_timers is using the perfmon config file, the
# perf_timer will be patched in only if perfmon is enabled.
#
# For now
# Their overhead if perfmon is disabled is incredibly minimal but not zero.
from ._timers import add_counter_event, add_instant_event, perf_timer, timers

# If perfmon is disabled then no functions are monkey patched and all
# perfmon functions are stubs that do basically nothing.
#
# Nevertheless for now manually added perf_timers, counter_events and
# instant_events should be commented out or removed before merging.
USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

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
#     Counter events appear as a little bar graph over time.
#
# add_instant_event
#     Instant events appear as a vertical line in the Chrome UI.
#
# The best way to add perf_timers is using the perfmon config file, the
# perf_timer will be patched in only if perfmon is enabled.
#
# Adding perf_timers "by hand" is sometimes helpful during intensive
# investigations, but consider them like "debug prints" something you
# strip out before commiting. When perfmon is disabled perf_timers
# do close to nothing, but there is still maybe 1 usec overhead.
from ._timers import add_counter_event, add_instant_event, perf_timer, timers

# If not using perfmon timers will be 100% disabled with hopefully zero
# run-time impact.
USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

"""Performance Monitoring.

Enable perfmon by defining NAPARI_PERFMON to be non-zero.

The best way to add perf_timers is using the perfmon config file. You can
list which methods or functions you want to time, and a perf_timer will be
monkey-patched into each callable on startup. The monkey patching
is done only if perfmon is enabled.

You can also use the "perf_timer" context object and "add_counter_event"
and "add_instant_event", but all three of these should be removed before
merging the PR into master. While they have almost zero overhead when
perfmon is disable, it will still result in empty function calls.
"""
import os

from ._compat import perf_counter_ns
from ._config import perf_config
from ._event import PerfEvent
from ._timers import add_counter_event, add_instant_event, perf_timer, timers

USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

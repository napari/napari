"""Performance Monitoring.

The perfmon module lets you instrument your code and visualize its run-time
behavior and timings in Chrome's Tracing GUI.

To enable perfmon define the env var NAPARI_PERFMON as follows:

NAPARI_PERFMON=1 
    Activates perfmon, trace using Debug -> Performance Trace menu.

NAPARI_PERFMON=/path/to/config.json
    Configure perfmon using the config.json configuration. See the
    PerfmonConfig docs for the spec of the config file.

Chrome Tracing
---------------

Chrome has a nice built-in performance tool called chrome://tracing. Chrome
can record traces of web applications. But the format is well-documented and
anyone can create the files and use the nice GUI. And other programs accept
the format including:
1) https://www.speedscope.app/ which does flamegraphs (Chrome doesn't).
2) Qt Creator's performance tools.

Monkey Patching
---------------

The best way to add perf_timers is using the perfmon config file. You can
list which methods or functions you want to time, and a perf_timer will be
monkey-patched into each callable on startup. The monkey patching
is done only if perfmon is enabled.

Trace On Start
---------------
Add a line to the config file like:

    "trace_file_on_start": "/Path/to/my/trace.json"

Perfmon will start tracing on startup. You must quit napari with the Quit
command for napari to write trace file. See PerfmonConfig docs.

Manual Timing
-------------

You can also manually add "perf_timer" context objects and
"add_counter_event()" and "add_instant_event()" functions to your code. All
three of these should be removed before merging the PR into master. While
they have almost zero overhead when perfmon is disabled, it's still better
not to leave them in the code. Think of them as similar to debug prints.
"""
import os

from ._compat import perf_counter_ns
from ._config import perf_config
from ._event import PerfEvent
from ._timers import (
    add_counter_event,
    add_instant_event,
    block_timer,
    perf_timer,
    timers,
)

USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

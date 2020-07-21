"""The perf_timer context object.
"""
import contextlib
import os
from typing import Optional

from ._compat import perf_counter_ns
from ._event import PerfEvent
from ._timers import timers

USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

if USE_PERFMON:

    @contextlib.contextmanager
    def perf_timer(
        name: str,
        category: Optional[str] = None,
        print_time: bool = False,
        **kwargs,
    ):
        """Time a block of code.

        Parameters
        ----------
        name : str
            The name of this timer.
        category : str
            Comma separated categories such has "render,update".
        **kwargs : dict
            Additional keyword arguments for the "args" field of the event.

        Examples
        --------
        with perf_timer("draw"):
            draw_stuff()
        """
        start_ns = perf_counter_ns()
        yield
        end_ns = perf_counter_ns()
        event = PerfEvent(name, start_ns, end_ns, **kwargs)
        timers.add_event(event)
        if print_time:
            ms = (end_ns - start_ns) / 1e6
            print(f"{name} {ms}ms")


else:
    # contextlib.nullcontext does not work with kwargs, so we just
    # create a do-nothing context object. This is not zero overhead
    # but it's very low, about 1 microsecond? But because it's not
    # zero it's best practice not to commit perf_timers, think of
    # them like debug prints.
    @contextlib.contextmanager
    def perf_timer(name: str, category: Optional[str] = None, **kwargs):
        yield

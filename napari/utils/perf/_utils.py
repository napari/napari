"""Performance monitoring utilities.

1) perf_timer contex manager times a block of code.
2) perf_func decorators time functions.
"""
import contextlib
import functools
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

    def perf_func(func):
        """Decorator to time a function.

        The timer name is automatically based on the function's name.

        Parameters
        ----------
        func
            The function we are wrapping.

        Examples
        --------
        @perf_func
        def draw(self):
            draw_stuff()
        """
        # Name is class.method
        timer_name = f"{func.__qualname__}"

        # Full name is included as an arg, so it does not clutter the main
        # timeline view. Viewable when you click on the event.
        full_name = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            with perf_timer(timer_name, function=full_name):
                return func(*args, **kwargs)

        return wrapper

    def perf_func_named(timer_name: str):
        """Decorator to time a function where we specify the timer name.

        Parameters
        ----------
        timer_name : str
            The name to give this timer.

        Examples
        --------
        @perf_func_name("important draw")
        def draw(self):
            draw_stuff()
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with perf_timer(timer_name, "decorator"):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


else:
    # contextlib.nullcontext does not work with kwargs?
    @contextlib.contextmanager
    def perf_timer(name: str, category: Optional[str] = None, **kwargs):
        yield

    def perf_func(func):
        return func

    def perf_func_named(timer_name: str):
        def decorator(func):
            return func

        return decorator

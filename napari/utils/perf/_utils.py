"""Utilities to support performance monitoring:

1) context manager: perf_timer times a block of code.
2) decorator: perf_func times a function.
"""
import contextlib
import functools
from typing import Optional

from ._compat import perf_counter_ns
from ._config import USE_PERFMON, PYTHON_3_7
from ._event import PerfEvent
from ._timers import timers


if USE_PERFMON:

    @contextlib.contextmanager
    def perf_timer(name: str, category: Optional[str] = None):
        """Time a block of code.

        Parameters
        ----------
        name : str
            The name of this timer.
        category : str, optional
            Category for this timer.

        Examples
        --------
        with perf_timer("draw"):
            draw_stuff()
        """
        start_ns = perf_counter_ns()
        yield
        end_ns = perf_counter_ns()
        event = PerfEvent(category, name, start_ns, end_ns)
        timers.add_event(event)

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
        # Name alone first so that's visible in the GUI first.
        timer_name = f"{func.__name__} - {func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def time_function(*args, **kwargs):

            with perf_timer(timer_name, "decorator"):
                return func(*args, **kwargs)

        return time_function

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
            def time_function(*args, **kwargs):
                with perf_timer(timer_name, "decorator"):
                    return func(*args, **kwargs)

            return time_function

        return decorator


else:
    # Disable both with hopefully zero run-time overhead.
    if PYTHON_3_7:
        perf_timer = contextlib.nullcontext()
    else:

        @contextlib.contextmanager
        def perf_timer(name: str):
            yield

    def perf_func():
        def decorator(func):
            return func

    def perf_func_named(timer_name: str):
        def decorator(func):
            return func

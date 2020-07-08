"""Performance monitoring utilities.

1) perf_timer contex manager times a block of code.
2) perf_func decorators time functions.
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

        Example
        -------
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

        Example
        -------
        @perf_func
        def draw(self):
            draw_stuff()
        """
        # Put bar name first so GUI shows that, then the full name.
        timer_name = f"{func.__name__} - {func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            with perf_timer(timer_name, "decorator"):
                return func(*args, **kwargs)

        return wrapper

    def perf_func_named(timer_name: str):
        """Decorator to time a function where we specify the timer name.

        Parameters
        ----------
        timer_name : str
            The name to give this timer.

        Example
        -------
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
    # Not using perfmon so disable the perf context object and the
    # decorators with hopefully negligible run-time overhead.
    if PYTHON_3_7:
        perf_timer = contextlib.nullcontext
    else:

        @contextlib.contextmanager
        def perf_timer(name: str, category: Optional[str] = None):
            yield

    def perf_func(func):
        return func

    def perf_func_named(timer_name: str):
        def decorator(func):
            return func

        return decorator

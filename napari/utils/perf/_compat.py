"""Compatibility functions.
"""
import time

from ._config import PYTHON_3_7


if PYTHON_3_7:
    # Use the real perf_counter_ns
    perf_counter_ns = time.perf_counter_ns
else:

    def perf_counter_ns():
        """Compatibility version for pre Python 3.7."""
        return int(time.perf_counter() * 1e9)

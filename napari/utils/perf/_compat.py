"""Perfmon related compatibility functions.
"""
import sys
import time

GREATER_EQUAL_PY37 = sys.version_info[:2] >= (3, 7)


def _perf_counter_ns():
    """Compatibility version for pre Python 3.7."""
    return int(time.perf_counter() * 1e9)


perf_counter_ns = (
    time.perf_counter_ns if GREATER_EQUAL_PY37 else _perf_counter_ns
)

"""Perfmon related compatibility functions.
"""
import sys
import time

PYTHON_3_7_OR_NEWER = sys.version_info[:2] >= (3, 7)

if PYTHON_3_7_OR_NEWER:
    # Use the real perf_counter_ns
    perf_counter_ns = time.perf_counter_ns
else:

    def perf_counter_ns():
        """Compatibility version for pre Python 3.7."""
        return int(time.perf_counter() * 1e9)

"""Perfmon related compatibility functions.
"""
import sys
import time

if sys.version_info[:2] >= (3, 7):
    # Use the real perf_counter_ns
    perf_counter_ns = time.perf_counter_ns
else:

    def perf_counter_ns():
        """Compatibility version for pre Python 3.7."""
        return int(time.perf_counter() * 1e9)

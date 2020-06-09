"""Perf configuration flags.
"""
import os
import sys

# If USE_PERFMON is not set then performance timers will be 100% disabled with
# hopefully zero run-time impact.
USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

# We have some pre-3.7 functionality.
PYTHON_3_7 = sys.version_info[:2] >= (3, 7)

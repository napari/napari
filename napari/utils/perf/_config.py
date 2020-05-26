"""Perf configuration flags.
"""
import os
import sys

# For now all performance timing is 100% disabled with hopefully zero run-time
# overhead unless the below environment variable is set.
USE_PERFMON = os.getenv("NAPARI_PERFMON", "0") != "0"

# We have some pre-3.7 functionality.
PYTHON_3_7 = sys.version_info[:2] >= (3, 7)

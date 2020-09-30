"""ChunkLoader and related.
"""
import os

from ._loader import chunk_loader
from ._request import ChunkKey, ChunkRequest

_async = os.getenv("NAPARI_ASYNC", "0") != "0"
_pytest = "PYTEST_CURRENT_TEST" in os.environ

# Nothing should be imported unless async is defined.
assert _async or _pytest

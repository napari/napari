"""ChunkLoader and related.
"""
import os

from ._loader import chunk_loader
from ._request import ChunkKey, ChunkRequest

_async = os.getenv("NAPARI_ASYNC", "0") != "0"
_pytest = _pytest = "pytest" in sys.modules

# Nothing should be imported unless async is defined or we are in pytest.
assert _async or _pytest

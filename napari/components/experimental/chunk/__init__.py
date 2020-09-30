"""ChunkLoader and related.
"""
import os

from ._loader import chunk_loader
from ._request import ChunkKey, ChunkRequest

# Nothing should be imported unless async is defined.
assert os.getenv("NAPARI_ASYNC", "0") != "0"

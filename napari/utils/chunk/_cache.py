"""ChunkCache stores loaded chunks.

NOTE: we might end up just using the Dask cache instead of our own
ChunkCache. The reason is we need the Dask cache for non-image layers,
but having 2 caches is not good. So using the Dask cache for everything
is the only obvious way to have on single cache.
"""
import logging
from typing import Dict, Optional

import numpy as np
from cachetools import LRUCache

from ._request import ChunkRequest

# ChunkCache size as a fraction of total RAM, needs to be configurable
# and even dynamically configurable.
CACHE_MEM_FRACTION = 0.1

LOGGER = logging.getLogger("ChunkLoader")

ChunkArrays = Dict[str, np.ndarray]


def _get_cache_size_bytes(mem_fraction):
    import psutil

    # Sizing approach borrowed from our create_dask_cache()
    return psutil.virtual_memory().total * mem_fraction


def _getsizeof_chunks(chunks: ChunkArrays) -> int:
    return sum(array.nbytes for array in chunks.values())


class ChunkCache:
    """Cache of previously loaded chunks.
    """

    def __init__(self):
        nbytes = _get_cache_size_bytes(CACHE_MEM_FRACTION)
        self.chunks = LRUCache(maxsize=nbytes, getsizeof=_getsizeof_chunks)

    def add_chunk(self, request: ChunkRequest) -> None:
        """Add this chunk to the cache.

        Parameters
        ----------
        request : ChunkRequest
            Add the data in this request to the cache.
        """
        LOGGER.info("ChunkCache.add_chunk: %s", request.key)
        self.chunks[request.key] = request.chunks

    def get_chunk(self, request: ChunkRequest) -> Optional[ChunkArrays]:
        """Get the cached data for this chunk request.

        TODO_ASYNC: assumes there's just one layer....
        """
        LOGGER.info("ChunkCache.get_chunk: %s", request.key)
        return self.chunks.get(request.key)

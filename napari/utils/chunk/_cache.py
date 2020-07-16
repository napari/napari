"""ChunkCache stores loaded chunks.

NOTE: we might end up just using the Dask cache even for chunks...
"""
import logging
from typing import Optional

import numpy as np
from cachetools import LRUCache

from ...types import ArrayLike

from ._request import ChunkRequest

# ChunkCache size as a fraction of total RAM, needs to be configurable
# and even dynamically configurable.
CACHE_MEM_FRACTION = 0.1

LOGGER = logging.getLogger("ChunkLoader")


def _get_cache_size_bytes(mem_fraction):
    import psutil

    # Sizing approach borrowed from our create_dask_cache()
    return psutil.virtual_memory().total * mem_fraction


def _getsizeof_chunk(array: np.ndarray):
    return array.nbytes


class ChunkCache:
    """Cache of previously loaded chunks.
    """

    def __init__(self):
        nbytes = _get_cache_size_bytes(CACHE_MEM_FRACTION)
        self.chunks = LRUCache(maxsize=nbytes, getsizeof=_getsizeof_chunk)

    def add_chunk(self, request: ChunkRequest) -> None:
        """Add this chunk to the cache.

        Parameters
        ----------
        request : ChunkRequest
            Add the data in this request to the cache.
        """
        LOGGER.info("ChunkCache.add_chunk: %s", request.key)
        self.chunks[request.key] = request.array

    def get_chunk(self, request: ChunkRequest) -> Optional[ArrayLike]:
        """Get the cached data for this chunk request.

        TODO_ASYNC: assumes there's just one layer....
        """
        LOGGER.info("ChunkCache.get_chunk: %s", request.key)
        return self.chunks.get(request.key)

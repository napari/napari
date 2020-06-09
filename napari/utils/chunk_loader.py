"""Asynchronously load chunks for rendering.
"""

from concurrent import futures

import numpy as np

from ..types import ArrayLike


def _load_chunk(array: ArrayLike) -> ArrayLike:
    return np.asarray(array)


class ChunkLoader:
    """Load chunks for rendering.
    """

    NUM_WORKER_THREADS = 1

    def __init__(self):
        self.executor = futures.ThreadPoolExecutor(
            max_workers=self.NUM_WORKER_THREADS
        )
        self.requests = []

    def request_chunk(self, array: ArrayLike):
        """Request that the loader load this chunk.

        array : ArrayLike
            Load data from this array-like object in a worker thread.
        """
        self.requests.append(array)
        future = self.executor.submit(_load_chunk, array)
        self.futures.append(future)

    def clear(self, array_like):
        # Clear pending requests not yet starter.
        # We cannot currently cancel load that are in progress.
        self.requests.clear()


CHUNK_LOADER = ChunkLoader()

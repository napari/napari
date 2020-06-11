"""Asynchronously load chunks for rendering.
"""

from concurrent import futures

import numpy as np
from qtpy.QtCore import Signal, QObject


from ..types import ArrayLike


class ChunkLoaderSignals(QObject):
    chunk_loaded = Signal()


class ChunkRequest:
    """Ask the ChunkLoader to load this data in a worker thread.

    Placeholder class: get rid of this class if it doesn't grow!

    Parameters
    ----------
    array : ArrayLike
        Load the data from this array.
    """

    def __init__(self, indices, array: ArrayLike, callback):
        self.indices = indices
        self.array = array
        self.callback = callback


def _chunk_loader_worker(request: ChunkRequest):
    request.array = np.asarray(request.array)
    return request


class ChunkLoader:
    """Load chunks for rendering.
    """

    NUM_WORKER_THREADS = 1
    signals = ChunkLoaderSignals()

    def __init__(self):
        self.executor = futures.ThreadPoolExecutor(
            max_workers=self.NUM_WORKER_THREADS
        )
        self.futures = []

    def load_chunk(self, request: ChunkRequest):
        """Request this just is loaded asynchronously.

        array : ArrayLike
            Load data from this array-like object in a worker thread.
        """
        print(f"load_chunk: {request.indices}")
        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self.done)
        self.futures.append(future)
        return future

    def done(self, future):
        request = future.result()
        print(f"done: {request.indices}")
        request.callback()
        self.signals.chunk_loaded.emit()

    def clear(self, array_like):
        # Clear pending requests not yet starter.
        # We cannot currently cancel load that are in progress.
        self.requests.clear()


CHUNK_LOADER = ChunkLoader()

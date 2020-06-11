"""ChunkLoader and related classes.
"""

from concurrent import futures
import time

import numpy as np
from qtpy.QtCore import Signal, QObject


from ..types import ArrayLike


class ChunkRequest:
    """A ChunkLoader request: please load this chunk.

    Placeholder class: I anticipate this class will grow, if not we can
    turn it into a namedtuple.

    Parameters
    ----------
    array : ArrayLike
        Load the data from this array.
    """

    def __init__(self, layer, indices, array: ArrayLike):
        self.layer = layer
        self.indices = indices
        self.array = array


def _chunk_loader_worker(request: ChunkRequest):
    """Worker thread load the array.

    This np.array() call might lead to IO or computation via dask or
    similar means which is why we are doing it in a worker thread!
    """
    request.array = np.asarray(request.array)
    time.sleep(1)
    return request


class ChunkLoaderSignals(QObject):
    chunk_loaded = Signal(ChunkRequest)


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
        print(f"ChunkLoader.done: {request.indices}")
        self.signals.chunk_loaded.emit(request)

    def clear(self, array_like):
        raise NotImplementedError()

    def remove_layer(self, layer):
        print(f"remove layer: {layer}")


CHUNK_LOADER = ChunkLoader()

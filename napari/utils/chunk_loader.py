"""ChunkLoader and related classes.
"""

from concurrent import futures
import time

import numpy as np
from qtpy.QtCore import Signal, QObject


from ..types import ArrayLike


def _index_to_tuple(index):
    """Slice is not hashable so we need a tuple.

    Parameters
    ----------
    index
        Could be a numeric index or a slice.
    """
    if isinstance(index, slice):
        return (index.start, index.stop, index.step)
    return index


class ChunkRequest:
    """A ChunkLoader request: please load this chunk.

    Placeholder class: I anticipate this class will grow, if not we can
    turn it into a namedtuple.

    Parameters
    ----------
    layer
        The layer requesting the data/
    indices
        The tuple of slices index into the data.
    array : ArrayLike
        Load the data from this array.
    """

    def __init__(self, layer, indices, array: ArrayLike):
        self.layer = layer
        self.indices = indices
        self.array = array

    @property
    def key(self):
        """Hashable key.

        Slice objects are not hashable.
        """
        return tuple(_index_to_tuple(x) for x in self.indices)


def _chunk_loader_worker(request: ChunkRequest):
    """Worker thread that loads the array.

    This np.array() call might lead to IO or computation via dask or
    similar means which is why we are doing it in a worker thread!
    """
    request.array = np.asarray(request.array)
    time.sleep(1)
    return request


class ChunkLoaderSignals(QObject):
    """QtViewer connects to this.

    We need to notify from a worker thread to the GUI thread so knows to
    use the chunk we just loaded. The only way to do that is with Qt
    signals/slots/events.
    """

    chunk_loaded = Signal(ChunkRequest)


class ChunkCache:
    """Cache of recently loaded chunks.

    TODO_ASYNC: need LRU eviction, sizing based on RAM...
    """

    def __init__(self):
        self.chunks = {}

    def add_chunk(self, request: ChunkRequest) -> None:
        """Add recently loaded chunk to the cache.
        """
        print(f"ChunkCache.add_chunk: {request.key}")
        self.chunks[request.key] = request.array

    def get_chunk(self, request: ChunkRequest):
        """Return chunk data or None if not found.

        TODO_ASYNC: assumes there's just one layer....
        """
        return self.chunks.get(request.key)


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
        self.cache = ChunkCache()

    def load_chunk(self, request: ChunkRequest):
        """Load this chunk asynchronously.

        Called from the GUI thread by Image or ImageSlice.

        request : ChunkRequest
            Contains the array to load from and related info.
        """
        print(f"load_chunk: {request.indices}")
        array = self.cache.get_chunk(request)

        if array is not None:
            print(f"load_chunk: cache hit {request.indices}")

            # Cache hit, request is satisfied.
            request.array = array
            return request
        print(f"load_chunk: cache miss {request.indices}")

        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self.done)
        self.futures.append(future)

        # Async load was started, nothing available yet.
        return None

    def done(self, future):
        """Future was done, success or cancelled.

        Called in the worker thread.
        """
        request = future.result()
        print(f"ChunkLoader.done: {request.indices}")

        # Do this from worker thread for now. It's safe for now.
        # TODO_ASYNC: Maybe switch to GUI thread but then we need an event.
        self.cache.add_chunk(request)

        # Notify QtViewer in the GUI thread, it will pass the data to the
        # layer that requested it.
        self.signals.chunk_loaded.emit(request)

    def clear_queued(self):
        """Clear queued but not yet started requests.

        We can't clear in-progress requests that are already running in the
        worker thread.
        """
        self.futures[:] = [x for x in self.futures if x.cancel()]

    def remove_layer(self, layer):
        print(f"remove layer: {layer}")


CHUNK_LOADER = ChunkLoader()

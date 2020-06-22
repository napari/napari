"""ChunkLoader and related classes.

There is one global CHUNK_LOADER instance to handle async loading for any
and all Viewer instances that are running. There are two main reasons we
just have one and not one per Viewer:

1. We size the ChunkCache as a fraction of RAM, so having more than one
   would use too much RAM in many cases.

2. We (will) size the thread pool for optimal performance, and having
   multiple pools would result in more threads than we want.
"""
from collections import defaultdict
from concurrent import futures
import logging
from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import Signal, QObject


from ..types import ArrayLike

LOGGER = logging.getLogger("ChunkLoader")

fh = logging.FileHandler('chunk_loader.log')
LOGGER.addHandler(fh)
LOGGER.setLevel(logging.INFO)


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

    Parameters
    ----------
    data_id
        Pythod id() for the Layer._data we are viewing.
    indices
        The tuple of slices index into the data.
    array : ArrayLike
        Load the data from this array.
    """

    def __init__(self, data_id, indices, array: ArrayLike):
        self.data_id = data_id
        self.indices = indices
        self.array = array

        # Slice objects are not hashable, so turn them into tuples.
        indices_tuple = tuple(_index_to_tuple(x) for x in self.indices)

        # Key is data_id + indices as a tuples.
        self.key = tuple([self.data_id, indices_tuple])


def _chunk_loader_worker(request: ChunkRequest):
    """Worker thread that loads the array.

    This np.array() call might lead to IO or computation via dask or
    similar means which is why we are doing it in a worker thread!
    """
    request.array = np.asarray(request.array)
    return request


class ChunkLoaderSignals(QObject):
    """QtViewer connects to this.

    We need to notify from a worker thread to the GUI thread so knows to
    use the chunk we just loaded. The only way to do that is with Qt
    signals/slots/events.

    TODO_ASYNC: Create a wrapper so we don't need to import Qt at all?
    """

    chunk_loaded = Signal(ChunkRequest)


class ChunkCache:
    """Cache of previously loaded chunks.

    TODO_ASYNC: need LRU eviction, sizing based on RAM, etc.
    """

    def __init__(self):
        self.chunks = {}

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


class ChunkLoader:
    """Load chunks for rendering.
    """

    NUM_WORKER_THREADS = 1
    signals = ChunkLoaderSignals()

    def __init__(self):
        self.executor = futures.ThreadPoolExecutor(
            max_workers=self.NUM_WORKER_THREADS
        )
        # Maps data_id to futures for that layer.
        self.futures: Dict[int, List[futures.Future]] = defaultdict(list)
        self.cache = ChunkCache()

    def load_chunk(self, request: ChunkRequest):
        """Load this chunk asynchronously.

        Called from the GUI thread by Image or ImageSlice.

        request : ChunkRequest
            Contains the array to load from and related info.
        """
        # Clear any existing futures. We only support non-multi-scale so far
        # and there can only be one load in progress per layer.
        self.clear_pending(request.data_id)

        LOGGER.info("ChunkLoader.load_chunk: %s", request.key)
        array = self.cache.get_chunk(request)

        if array is not None:
            LOGGER.info("load_chunk: cache hit %s", request.key)

            # Cache hit, request is satisfied.
            request.array = array
            return request

        LOGGER.info("ChunkLoader.load_chunk: cache miss %s", request.key)

        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self.done)

        # Future is in progress for this layer.
        self.futures[request.data_id].append(future)

        # Async load was started, nothing is available yet.
        return None

    def done(self, future):
        """Future was done, success or cancelled.

        Called in the worker thread.
        """
        request = future.result()
        LOGGER.info("ChunkLoader.done: %s", request.key)

        # Do this from worker thread for now. It's safe for now.
        # TODO_ASYNC: Maybe switch to GUI thread but then we need an event.
        self.cache.add_chunk(request)

        # Notify QtViewer in the GUI thread, it will pass the data to the
        # layer that requested it.
        self.signals.chunk_loaded.emit(request)

    def clear_pending(self, data_id: int) -> None:
        """Clear any pending requests for this data_id.

        We can't clear in-progress requests that are already running in the
        worker thread, which is too bad.
        """
        future_list = self.futures[data_id]

        # Try to clear them all. If cancel() returns false it mean the
        # future is running and we can't cancel it.
        num_before = len(future_list)
        future_list[:] = [x for x in future_list if x.cancel()]
        num_after = len(future_list)
        num_cleared = num_before - num_after

        if num_before == 0:
            LOGGER.info("ChunkLoader.clear_layer: empty")
        else:
            LOGGER.info(
                "ChunkLoader.clear_layer: %d of %d cleared -> %d remain",
                num_cleared,
                num_before,
                num_after,
            )

    def remove_layer(self, layer) -> None:
        LOGGER.info("ChunkLoader.remove_layer: %s", id(layer))


CHUNK_LOADER = ChunkLoader()

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
from contextlib import contextmanager
from concurrent import futures
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qtpy.QtCore import Signal, QObject


from ..types import ArrayLike

LOGGER = logging.getLogger("ChunkLoader")

fh = logging.FileHandler('chunk_loader.log')
LOGGER.addHandler(fh)
LOGGER.setLevel(logging.INFO)

# We convert slices to tuple for hashing.
SliceTuple = Tuple[int, int, int]


def _index_to_tuple(index: Union[int, slice]) -> Union[int, SliceTuple]:
    """Get hashable object for the given index.

    Slice is not hashable so we convert slices to tuples.

    Parameters
    ----------
    index
        Integer index or a slice.

    Returns
    -------
    Union[int, SliceTuple]
        Hashable object that can be used for the index.
    """
    if isinstance(index, slice):
        return (index.start, index.stop, index.step)
    return index


def _get_synchronous() -> bool:
    """
    Return True if ChunkManager should load data synchronously.

    Returns
    -------
    bool
        True if loading should be synchronous.
    """
    # Async is off by default for now. Must opt-in with NAPARI_ASYNC_LOAD.
    synchronous_loading = True

    env_var = os.getenv("NAPARI_ASYNC_LOAD")

    if env_var is not None:
        # Overide the deafult with the env var's setting.
        synchronous_loading = env_var == "0"

    return synchronous_loading


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

    def __init__(self, layer, data_id, indices, array: ArrayLike):
        self.layer = layer
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

    # If loading is synchronous then the ChunkLoader is essentially
    # disabled, load_chunk() will immediately do the load in the GUI
    # thread, then it will return the satisfied request.
    synchronous = _get_synchronous()

    def __init__(self):
        self.executor = futures.ThreadPoolExecutor(
            max_workers=self.NUM_WORKER_THREADS
        )
        # Maps data_id to futures for that layer.
        self.futures: Dict[int, List[futures.Future]] = defaultdict(list)
        self.cache = ChunkCache()

    def load_chunk(self, request: ChunkRequest) -> Optional[ChunkRequest]:
        """Load the array in the given ChunkRequest.

        If ChunkLoader.synchronous_loading is set the load is performed
        immediately in the GUI thread and the satisfied request is returned.

        Otherwise an asynchronous load is requested and None is returned.
        The load will be performed in a worker thread. Later Layer.chunk_loaded()
        will be called in the GUI thread.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.

        Optional[ChunkRequest]
            The satisfied ChunkRequest or None indicating an async load.
        """
        if ChunkLoader.synchronous:
            # Load it immediately right here in the GUI thread.
            request.array = np.asarray(request.array)
            return request

        self._load_async(request)
        return None

    def _load_async(self, request: ChunkRequest) -> None:
        """Initiate an asynchronous load of the given request.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.
        """
        LOGGER.info("ChunkLoader._load_async: %s", request.key)

        # Clear any existing futures for this specific data_id. We only
        # support non-multi-scale so far and there can only be one load in
        # progress per layer.
        self._clear_pending(request.data_id)

        # Check the cache first.
        array = self.cache.get_chunk(request)

        if array is not None:
            LOGGER.info("ChunkLoader._load_async: cache hit %s", request.key)
            request.array = array
            return request

        LOGGER.info("ChunkLoader.load_chunk: cache miss %s", request.key)

        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self._done)

        # Future is in progress for this layer.
        self.futures[request.data_id].append(future)

        # Async load was started, nothing is available yet.
        return None

    def _done(self, future: futures.Future) -> None:
        """Future finished with success or was cancelled.

        This is called from the worker thread.
        """
        request = future.result()
        LOGGER.info("ChunkLoader._done: %s", request.key)

        # Do this from worker thread for now. It's safe for now.
        # TODO_ASYNC: Ultimately we might want to this to happen from the
        # GUI thread so all cache access is from the same thread.
        self.cache.add_chunk(request)

        # Notify QtViewer in the GUI thread, it will pass the data to the
        # layer that requested it.
        self.signals.chunk_loaded.emit(request)

    def _clear_pending(self, data_id: int) -> None:
        """Clear any pending requests for this data_id.

        We can't clear in-progress requests that are already running in the
        worker thread. This is too bad since subsequent requests might have
        to wait behind them. Terminating threads is considered unsafe.

        Long term we could maybe allow the user to create special
        "cancellable" tasks or dask-arrays somehow, if they periodically
        checked a flag and gracefully exited. They would perform slightly
        better then opaque non-cancellable task.
        """
        future_list = self.futures[data_id]

        # Try to cancel them all, cancel() will return false if running.
        num_before = len(future_list)
        future_list[:] = [x for x in future_list if x.cancel()]
        num_after = len(future_list)
        num_cleared = num_before - num_after

        if num_before == 0:
            LOGGER.info("ChunkLoader.clear_pending: empty")
        else:
            LOGGER.info(
                "ChunkLoader.clear_pending: %d of %d cleared -> %d remain",
                num_cleared,
                num_before,
                num_after,
            )

    def remove_layer(self, layer) -> None:
        LOGGER.info("ChunkLoader.remove_layer: %s", id(layer))


@contextmanager
def synchronous_loading():
    """Context object to temporarily disable async loading.

    with synchronous_loading():
        layer = Image(data)
        ... use layer ...
    """
    previous = ChunkLoader.synchronous
    ChunkLoader.synchronous = True
    yield
    ChunkLoader.synchronous = previous


CHUNK_LOADER = ChunkLoader()

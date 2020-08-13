"""ChunkLoader and the synchronous_loading context object.

There is one global chunk_loader instance to handle async loading for all
Viewer instances. There are two main reasons we do this instead of one
ChunkLoader per Viewer:

1. We size the ChunkCache as a fraction of RAM, so having more than one
   cache would use too much RAM.

2. We might size the thread pool for optimal performance, and having
   multiple pools would result in more workers than we want.

Think of the ChunkLoader as a shared resource like "the filesystem" where
multiple clients can be access it at the same time, but it is the interface
to just one physical resource.
"""
import logging
import weakref
from concurrent import futures
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import dask.array as da
import numpy as np

from ...types import ArrayLike
from ...utils.event import EmitterGroup
from ..perf import perf_timer
from ._cache import ChunkCache
from ._config import async_config
from ._delay_queue import DelayQueue
from ._request import ChunkKey, ChunkRequest

LOGGER = logging.getLogger("ChunkLoader")

# Executor for either a thread pool or a process pool.
PoolExecutor = Union[futures.ThreadPoolExecutor, futures.ProcessPoolExecutor]


class StatWindow:
    """Maintain an average value over some rolling window.

    Adding values is very efficient (once the window is full) but
    calculating the average is O(size), although using numpy.

    Parameters
    ----------
    size : int
        The size of the window.
    """

    def __init__(self, size: int):
        self.size = size
        self.values = np.array([])  # float64 array
        self.index = 0  # insert values here once full

    def add(self, value: float):
        """Add one value to the window.

        Parameters
        ----------
        value : float
            Add this value to the window.
        """
        if len(self.values) < self.size:
            # This isn't super efficient but we're optimizing for the case
            # when the array is full and we are just poking in values.
            self.values = np.append(self.values, value)
        else:
            # Array is full, rotate through poking in one value at a time,
            # this should be very fast.
            self.values[self.index] = value
            self.index = (self.index + 1) % self.size

    @property
    def average(self):
        """Return the average of all values in the window."""
        if len(self.values) == 0:
            return 0  # Just say zero, really there is no value.
        return np.average(self.values)


def _chunk_loader_worker(request: ChunkRequest):
    """This is the worker thread or process that loads the array.

    We have workers because when we call np.asarray() that might lead
    to IO or computation which could take a while. We do not want to
    do that in the GUI thread or the UI will block.

    Parameters
    ----------
    request : ChunkRequest
        The request to load.
    """
    request.load_chunks()  # loads all chunks in the request
    return request


def _create_executor(use_processes: bool, num_workers: int) -> PoolExecutor:
    """Return the thread or process pool executor.

    Parameters
    ----------
    use_processes : bool
        If True use processes, otherwise threads.
    num_workers : int
        The number of worker threads or processes.
    """
    if use_processes:
        return futures.ProcessPoolExecutor(max_workers=num_workers)
    return futures.ThreadPoolExecutor(max_workers=num_workers)


def _get_type_str(data) -> str:
    """Get human readable name for the data's type.

    Returns
    -------
    str
        A string like "ndarray" or "dask".
    """
    data_type = type(data)

    if data_type == list:
        if len(data) == 0:
            return "EMPTY"
        else:
            # Recursively get the type string of the zeroth level.
            return _get_type_str(data[0])

    if data_type == da.Array:
        # Special case this because otherwise data_type.__name__
        # below would just return "Array".
        return "dask"

    # For class numpy.ndarray this returns "ndarray"
    return data_type.__name__


class LayerInfo:
    """Information about one layer we are tracking.

    Parameters
    ----------
    layer : Layer
        The layer we are loading chunks for.
    """

    # Window size for timing statistics. We use a simple average over the
    # window. This is better than the just "last load time" because it
    # won't jump around from one fluke. But we could do something much
    # fancier in the future with filtering.
    WINDOW_SIZE = 10

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)
        self.data_type: str = _get_type_str(layer.data)

        self.num_loads: int = 0
        self.num_chunks: int = 0
        self.num_bytes: int = 0

        # Keep running average of load times.
        self.load_time_ms: StatWindow = StatWindow(self.WINDOW_SIZE)

    def get_layer(self):
        """Resolve our weakref to get the layer, log if not found.

        Returns
        -------
        layer : Layer
            The layer for this ChunkRequest.
        """
        layer = self.layer_ref()
        if layer is None:
            LOGGER.info(
                "LayerInfo.get_layer: layer %d was deleted", self.layer_id
            )
        return layer

    def load_finished(self, request: ChunkRequest) -> None:
        """This ChunkRequest was satisfied, record stats.

        Parameters
        ----------
        request : ChunkRequest
            Record stats related to loading these chunks.
        """
        # Record the number of loads and chunks.
        self.num_loads += 1
        self.num_chunks += request.num_chunks

        # Total bytes loaded.
        self.num_bytes += request.num_bytes

        # Record the load time.
        load_ms = request.timers['load_chunks'].duration_ms
        self.load_time_ms.add(load_ms)


class ChunkLoader:
    """Loads chunks in worker threads or processes.

    We cannot call np.asarray() in the GUI thread because it might block on
    IO or a computation. So the ChunkLoader calls np.asarry() for us in
    a worker thread or process.

    Attributes
    ----------
    synchronous : bool
        If True the ChunkLoader is essentially disabled, loads are done
        synchronously in the GUI thread. If False loads are done
        asynchronously in a worker thread.
    load_seconds : float
        Sleep this long in the worker during a load.
    executor : Union[ThreadPoolExecutor, ProcessPoolExecutor]
        Our thread or process pool executor.
    futures : FutureMap
        In progress futures for each layer (data_id).
    cache : ChunkCache
        Cache of previously loaded chunks.
    events : EmitterGroup
        We only signal one event: chunk_loaded.
    """

    FutureMap = Dict[int, List[futures.Future]]
    LayerMap = Dict[int, LayerInfo]

    def __init__(self):
        self.synchronous: bool = async_config.synchronous
        self.load_seconds: float = async_config.load_seconds

        use_processes: bool = async_config.use_processes
        num_workers: int = async_config.num_workers

        LOGGER.info(
            "ChunkLoader.__init__ synchronous=%d processes=%d num_workers=%d",
            self.synchronous,
            use_processes,
            num_workers,
        )

        # Executor to our concurrent.futures pool for workers.
        self.executor: PoolExecutor = _create_executor(
            use_processes, num_workers
        )

        # Delay queue prevents us from spamming the worker pool when the
        # user is rapidly scrolling through slices.
        self.delay_queue = DelayQueue(
            async_config.delay_seconds, self._submit_async
        )

        # Maps data_id to futures for that layer.
        self.futures: self.FutureMap = {}
        self.cache: ChunkCache = ChunkCache()

        # We emit only one event:
        #     chunk_loaded - a chunk was loaded
        self.events = EmitterGroup(
            source=self, auto_connect=True, chunk_loaded=None
        )

        # We track information about each layer including a weak reference
        # to the layer. The ChunkRequest cannot hold the weak reference
        # itself because it won't serialize if we are using proceses.
        self.layer_map: self.LayerMap = {}

    def get_info(self, layer_id: int) -> Optional[LayerInfo]:
        """Get LayerInfo for this layer or None."""
        return self.layer_map.get(layer_id)

    def create_request(
        self, layer, key: ChunkKey, chunks: Dict[str, ArrayLike]
    ):
        """Create a ChunkRequest for submission to load_chunk.

        Parameters
        ----------
        layer : Layer
            The layer that's requesting the data.
        indices : slice
            The indices being requested.
        array : ArrayLike
            The array containing the data we want to load.
        """
        layer_id = key.layer_id

        # Add a LayerInfo if we don't already have one.
        if layer_id not in self.layer_map:
            self.layer_map[layer_id] = LayerInfo(layer)

        # Return the new request.
        return ChunkRequest(key, chunks)

    def _asarray(self, array):
        """Get the array data. We break this out for perfmon timing.
        """
        return np.asarray(array)

    def load_chunk(self, request: ChunkRequest) -> Optional[ChunkRequest]:
        """Process the given ChunkRequest, load its data.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.

        ChunkRequest, optional
            The satisfied ChunkRequest or None indicating an async load
            was initiated

        Returns
        -------
        ChunkRequest, optional
            The ChunkRequest if it was satisfied otherwise None.

        Notes
        -----
        load_chunk() runs synchronously if ANY of these three things are true:

        1) ChunkLoader.synchronous is True. This means every load is done
           synchronously. As if async loads were never implemented.

        2) The request result was found in the ChunkCache.

        3) The request.in_memory value is True meaning the arrays are all
           ndarrays and so there no point in using a worker.

        If the load is synchronous then load_chunk() returns the now-satisfied
        request. Otherwise load_chunk() returns None and the Layer's
        chunk_loaded() will be called in the GUI thread sometime in the future
        after the worker as performed the load.
        """
        # We do an immediate load in the GUI thread in synchronous mode or
        # the request is already in memory (ndarrays). However if
        # self.load_seconds > 0 then we cannot load in the GUI thread
        # because we have to sleep that long in the worker.
        if self.synchronous or request.in_memory and not self.load_seconds:
            LOGGER.info("ChunkLoader.load_chunk")
            request.load_chunks_gui()
            return request

        # Check the cache first.
        chunks = self.cache.get_chunks(request)

        if chunks is not None:
            LOGGER.info("ChunkLoader._load_async: cache hit %s", request.key)
            request.chunks = chunks
            return request

        LOGGER.info("ChunkLoader.load_chunk: cache miss %s", request.key)

        # Clear any pending requests for this specific data_id. We don't
        # support "tiles" so there can be only one load in progress per
        # data_id.
        with perf_timer("clear_pending"):
            self._clear_pending(request.key.data_id)

        # This will force the work to sleep while processing this request.
        request.load_seconds = self.load_seconds

        # Add to the delay queue, the delay queue will
        # ChunkLoader_submit_async() later on if the delay expires without
        # the request getting cancelled.
        self.delay_queue.add(request)

    def _submit_async(self, request: ChunkRequest) -> None:
        """Initiate an asynchronous load of the given request.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.
        """
        LOGGER.info("ChunkLoader._load_async: %s", request.key)

        # Submit the future, have it call ChunkLoader._done when done.
        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self._done)

        # Store the future in case we need to cancel it.
        self.futures.setdefault(request.key.data_id, []).append(future)

    def _clear_pending(self, data_id: int) -> None:
        """Clear any pending requests for this data_id.

        Parameters
        ----------
        data_id : int
            Clear all requests associated with this data_id.
        """
        LOGGER.info("ChunkLoader._clear_pending %d", data_id)

        # Clear delay queue first. This are trivial to clear because they
        # have not even been submitted to the worker pool.
        self.delay_queue.clear(data_id)

        # Next get any futures for this data_id that have been submitted to
        # the pool.
        future_list = self.futures.setdefault(data_id, [])

        # Try to cancel all futures in the list, but cancel() will return
        # False if the task already started running. We cannot cancel tasks
        # that are already running, just have to let them run to
        # compeletion even though we probably do not care about what they
        # are loading.
        num_before = len(future_list)
        future_list[:] = [x for x in future_list if x.cancel()]
        num_after = len(future_list)
        num_cleared = num_before - num_after

        # Delete the list entirely if empty
        if num_after == 0:
            del self.futures[data_id]

        # Log what we did.
        if num_before == 0:
            LOGGER.info("ChunkLoader.clear_pending: empty")
        else:
            LOGGER.info(
                "ChunkLoader.clear_pending: %d of %d cleared -> %d remain",
                num_cleared,
                num_before,
                num_after,
            )

    @staticmethod
    def _get_request(future: futures.Future) -> Optional[ChunkRequest]:
        """Return the ChunkRequest for this future.

        Parameters
        ----------
        future futures.Future
            Get the request from this future.

        Returns
        -------
        Optional[ChunkRequest]
            The ChunkRequest or None if the future was cancelled.
        """
        try:
            # Our future has already finished since this is being
            # called from Chunk_Request._done(), so result() will
            # not block.
            return future.result()
        except futures.CancelledError:
            LOGGER.info("ChunkLoader._done: cancelled")
            return None

    def _done(self, future: futures.Future) -> None:
        """Called when a future finishes with success or was cancelled.

        Parameters
        ----------
        future : futures.Future
            The future that finished or was cancelled.

        Notes
        -----
        This method may be called in the worker thread. The documentation
        very intentionally does not specify which thread the future done
        callback will be called in, only that it will be called in the same
        process. So far on MacOS at least it seems to always be called in
        the worker thread when using a thread pool.
        """
        request = self._get_request(future)

        if request is None:
            return  # Future was cancelled, nothing to do.

        LOGGER.info("ChunkLoader._done: %s", request.key)

        # Now that we are back in this process we can add the perf events.
        # If we were using threads we could have just added the perf events
        # normally, but we do it this way just to keep it consistent, and
        # it should work just as well.
        request.add_perf_events()

        # Add chunks to the cache in the worker thread. For now it's safe
        # to do this in the worker. Later we might need to arrange for this
        # to be done in the GUI thread if cache access becomes more
        # complicated.
        self.cache.add_chunks(request)

        # Lookup this Request's LayerInfo.
        info = self._get_layer_info(request)

        if info is None:
            return  # Ignore the chunks since no LayerInfo.

        # Resolve the weakref.
        layer = info.get_layer()

        if layer is None:
            return  # Ignore chunks since layer was deleted.

        # Record LayerInfo stats.
        info.load_finished(request)

        # Fire event to tell QtChunkReceiver to forward this chunk to its
        # layer in the GUI thread.
        self.events.chunk_loaded(layer=layer, request=request)

    def _get_layer_info(self, request: ChunkRequest) -> Optional[LayerInfo]:
        """Return LayerInfo associated with this request or None.

        Parameters
        ----------
        request : ChunkRequest
            Return Layer_info for this request.
        """
        layer_id = request.key.layer_id

        try:
            return self.layer_map[layer_id]
        except KeyError:
            LOGGER.warn("ChunkLoader._done: no layer_id %d", layer_id)
            return None

    def wait(self, data_id: int) -> None:
        """Wait for the given data to be loaded.

        TODO_ASYNC: We do not use this today, but it could be useful.

        Parameters
        ----------
        data_id : int
            Wait on chunks for this data_id.
        """
        try:
            future_list = self.futures[data_id]
        except KeyError:
            LOGGER.warn("ChunkLoader.wait: no futures for data_id %d", data_id)
            return

        LOGGER.info(
            "ChunkLoader.wait: waiting on %d futures for %d",
            len(future_list),
            data_id,
        )

        # Call result() will block until the future has finished or was cancelled.
        [future.result() for future in future_list]
        del self.futures[data_id]

    def on_layer_deleted(self, layer):
        """The layer was deleted, delete it from our map.

        Layer
            The layer that was deleted.
        """
        try:
            del self.layer_map[id(layer)]
        except KeyError:
            # No chunk request was made for this layer
            pass


@contextmanager
def synchronous_loading(enabled):
    """Context object to temporarily disable async loading.

    with synchronous_loading(True):
        layer = Image(data)
        ... use layer ...
    """
    previous = chunk_loader.synchronous
    chunk_loader.synchronous = enabled
    yield
    chunk_loader.synchronous = previous


# Global instance
chunk_loader = ChunkLoader()

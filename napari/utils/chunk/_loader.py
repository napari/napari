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
from concurrent import futures
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import numpy as np

from ...types import ArrayLike
from ...utils.event import EmitterGroup
from ..perf import perf_timer
from ._cache import ChunkCache
from ._config import async_config
from ._delay_queue import DelayQueue
from ._info import LayerInfo, LoadType
from ._request import ChunkKey, ChunkRequest

LOGGER = logging.getLogger("ChunkLoader")

# Executor for either a thread pool or a process pool.
PoolExecutor = Union[futures.ThreadPoolExecutor, futures.ProcessPoolExecutor]


def _chunk_loader_worker(request: ChunkRequest) -> ChunkRequest:
    """This is the worker thread or process that loads the array.

    We call np.asarray() in a worker because it might lead to IO or
    computation which would block the GUI thread.

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


class ChunkLoader:
    """Loads chunks in worker threads or processes.

    We cannot call np.asarray() in the GUI thread because it might block on
    IO or a computation. So the ChunkLoader calls np.asarray() for us in
    a worker thread or process.

    Attributes
    ----------
    synchronous : bool
        If True LoadType.AUTO layers are loading synchronously in the GUI thread.
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
    ) -> ChunkRequest:
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

        if self._load_synchronously(request):
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

    def _load_synchronously(self, request: ChunkRequest) -> bool:
        """Return True if we should load this request synchronously."""

        info = self._get_layer_info(request)
        sync = self._do_sync_load(request, info)

        if not sync:
            return False  # We'll load it async.

        request.load_chunks()
        info.load_finished(request, sync=True)
        return True  # Load was sync.

    def _do_sync_load(self, request: ChunkRequest, info: LayerInfo) -> bool:
        """Return True if thislayer should load synchronously.

        Parameters
        ----------
        request : ChunkRequest
            The request we are loading.
        info : LayerInfo
            The layer we are loading the chunk into.
        """
        if info.load_type == LoadType.SYNC:
            return True  # Layer is forcing sync loads.

        if info.load_type == LoadType.ASYNC:
            return False  # Layer is forcing async loads.

        assert info.load_type == LoadType.AUTO  # AUTO is the only other type.

        # If ChunkLoader is synchronous then AUTO always means synchronous.
        if self.synchronous:
            return True

        # We need to sleep() so must be async.
        if self.load_seconds > 0:
            return False

        # If it's been loading "fast" then load synchronously. There's no
        # point doing async loading if it's just as easy to load immediately.
        if info.loads_fast:
            return True

        # Finally, load synchronously if it's an ndarray (in memory) otherwise
        # it's Dask or something else and we load async.
        return request.in_memory

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

        # Resolve the weakref.
        layer = info.get_layer()

        if layer is None:
            return  # Ignore chunks since layer was deleted.

        # Record LayerInfo stats.
        info.load_finished(request, sync=False)

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

        # Go ahead and raise KeyError if not found. This should never
        # happen because we add the layer to the layer_map in
        # ChunkLoader.create_request().
        return self.layer_map[layer_id]

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

"""ChunkLoader class.
"""
import logging
import os
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

from ....types import ArrayLike
from ....utils.config import octree_config
from ....utils.events import EmitterGroup
from ._cache import ChunkCache
from ._delay_queue import DelayQueue
from ._info import LayerInfo, LayerRef, LoadType
from ._request import ChunkKey, ChunkRequest

LOGGER = logging.getLogger("napari.async")

# Executor for either a thread pool or a process pool.
PoolExecutor = Union[ThreadPoolExecutor, ProcessPoolExecutor]


def _is_enabled(env_var) -> bool:
    """Return True if env_var is defined and not zero."""
    return os.getenv(env_var, "0") != "0"


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
        LOGGER.debug("ChunkLoader process pool num_workers=%d", num_workers)
        return ProcessPoolExecutor(max_workers=num_workers)

    LOGGER.debug("ChunkLoader thread pool num_workers=%d", num_workers)
    return ThreadPoolExecutor(max_workers=num_workers)


class ChunkLoader:
    """Loads chunks synchronously or asynchronously in worker thread or processes.

    We cannot call np.asarray() in the GUI thread because it might block on
    IO or a computation. So the ChunkLoader calls np.asarray() in a worker
    if doing async loading.

    Attributes
    ----------
    synchronous : bool
        If True all requests are loaded synchronously.
    num_workers : int
        The number of workers.
    executor : PoolExecutor
        The thread or process pool executor.
    futures : Dict[int, List[Future]]
        In progress futures for each layer (data_id).
    layer_map : Dict[int, LayerInfo]
        Stores a LayerInfo about each layer we are tracking.
    cache : ChunkCache
        Cache of previously loaded chunks.
    delay_queue : DelayQueue
        Requests sit in here for a bit before submission.
    events : EmitterGroup
        We only signal one event: chunk_loaded.
    """

    def __init__(self):
        config = octree_config['loader']
        self.force_synchronous: bool = bool(config['force_synchronous'])
        self.num_workers: int = int(config['num_workers'])
        self.use_processes: bool = bool(config['use_processes'])

        self.executor: PoolExecutor = _create_executor(
            self.use_processes, self.num_workers
        )

        self.futures: Dict[int, List[Future]] = {}
        self.layer_map: Dict[int, LayerInfo] = {}
        self.cache: ChunkCache = ChunkCache()

        # The DelayeQueue prevents us from spamming the worker pool when
        # the user is rapidly scrolling through slices.
        self.delay_queue = DelayQueue(
            config['delay_queue_ms'], self._submit_async
        )

        self.events = EmitterGroup(
            source=self, auto_connect=True, chunk_loaded=None
        )

    def get_info(self, layer_id: int) -> Optional[LayerInfo]:
        """Get LayerInfo for this layer or None."""
        return self.layer_map.get(layer_id)

    def create_request(
        self, layer_ref: LayerRef, key: ChunkKey, chunks: Dict[str, ArrayLike]
    ) -> ChunkRequest:
        """Create a ChunkRequest for submission to load_chunk.

        Parameters
        ----------
        layer_ref : LayerRef
            Reference to the layer that's requesting the data.
        key : ChunkKey
            The key for the request.
        chunks : Dict[str, ArrayLike]
            The arrays we want to load.
        """
        layer_id = layer_ref.layer_key.layer_id

        if layer_id not in self.layer_map:
            self.layer_map[layer_id] = LayerInfo(layer_ref)

        # Return the new request.
        return ChunkRequest(key, chunks)

    def load_chunk(self, request: ChunkRequest) -> Optional[ChunkRequest]:
        """Load the given request sync or async.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array or arrays to load.

        Returns
        -------
        Optional[ChunkRequest]
            The ChunkRequest if it was satisfied otherwise None.

        Notes
        -----
        We return a ChunkRequest if we performed the load synchronously,
        otherwise we return None is indicating that an asynchronous load
        was intitiated. When the async load finishes the layer's
        on_chunk_loaded() will be called from the GUI thread.
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

        # Clear any pending requests for this specific data_id.
        # TODO_OCTREE: turn this off because all our request come from the
        # same data_id. But maybe we can clear pending on something more
        # specific?
        # self._clear_pending(request.key.data_id)

        # Add to the delay queue, the delay queue will call our
        # _submit_async() method later on if the delay expires without the
        # request getting cancelled.
        self.delay_queue.add(request)
        return None

    def _load_synchronously(self, request: ChunkRequest) -> bool:
        """Return True if we loaded the request synchronously."""
        info = self._get_layer_info(request)

        if self._should_load_sync(request, info):
            request.load_chunks()
            info.stats.on_load_finished(request, sync=True)
            return True

        return False

    def _should_load_sync(
        self, request: ChunkRequest, info: LayerInfo
    ) -> bool:
        """Return True if this layer should load synchronously.

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

        # If forcing synchronous then AUTO always means synchronous.
        if self.force_synchronous:
            return True

        # If it's been loading "fast" then load synchronously. There's no
        # point is loading async if it loads really fast.
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
            Contains the arrays to load.
        """
        # Note about string formatting with logging: it's recommended to
        # use the oldest style of string formatting with logging. With
        # f-strings you'd pay the price of formatting the string even if
        # the log statement is disabled due to the log level, etc. In our
        # case the log will almost always be disabled unless debugging.
        # https://docs.python.org/3/howto/logging.html#optimization
        # https://blog.pilosus.org/posts/2020/01/24/python-f-strings-in-logging/
        LOGGER.debug("ChunkLoader._submit_async: %s", request.key)

        # Submit the future, have it call ChunkLoader._done when done.
        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self._done)

        # Store the future in case we need to cancel it.
        self.futures.setdefault(request.data_id, []).append(future)

    def _clear_pending(self, data_id: int) -> None:
        """Clear any pending requests for this data_id.

        Parameters
        ----------
        data_id : int
            Clear all requests associated with this data_id.
        """
        LOGGER.debug("ChunkLoader._clear_pending %d", data_id)

        # Clear delay queue first. These requests are trivial to clear
        # because they have not even been submitted to the worker pool.
        self.delay_queue.clear(data_id)

        # Get list of futures we submitted to the pool.
        future_list = self.futures.setdefault(data_id, [])

        # Try to cancel all futures in the list, but cancel() will return
        # False if the task already started running.
        num_before = len(future_list)
        future_list[:] = [x for x in future_list if x.cancel()]
        num_after = len(future_list)
        num_cleared = num_before - num_after

        # Delete the list entirely if empty
        if num_after == 0:
            del self.futures[data_id]

        # Log what we did.
        if num_before == 0:
            LOGGER.debug("ChunkLoader.clear_pending: empty")
        else:
            LOGGER.debug(
                "ChunkLoader.clear_pending: %d of %d cleared -> %d remain",
                num_cleared,
                num_before,
                num_after,
            )

    @staticmethod
    def _get_request(future: Future) -> Optional[ChunkRequest]:
        """Return the ChunkRequest for this future.

        Parameters
        ----------
        future : Future
            Get the request from this future.

        Returns
        -------
        Optional[ChunkRequest]
            The ChunkRequest or None if the future was cancelled.
        """
        try:
            # Our future has already finished since this is being
            # called from Chunk_Request._done(), so result() will
            # never block.
            return future.result()
        except CancelledError:
            LOGGER.debug("ChunkLoader._done: cancelled")
            return None

    def _done(self, future: Future) -> None:
        """Called when a future finishes with success or was cancelled.

        Parameters
        ----------
        future : Future
            The future that finished or was cancelled.

        Notes
        -----
        This method may be called in a worker thread. The
        concurrent.futures documentation very intentionally does not
        specify which thread the future's done callback will be called in,
        only that it will be called in some thread in the current process.
        """
        try:
            request = self._get_request(future)
        except ValueError:
            return  # Pool not running, app exit in progress.

        if request is None:
            return  # Future was cancelled, nothing to do.

        LOGGER.debug("ChunkLoader._done: %s", request.key)

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

        info.stats.on_load_finished(request, sync=False)

        # Fire event to tell QtChunkReceiver to forward this chunk to its
        # layer in the GUI thread.
        self.events.chunk_loaded(layer=layer, request=request)

    def _get_layer_info(self, request: ChunkRequest) -> LayerInfo:
        """Return LayerInfo associated with this request or None.

        Parameters
        ----------
        request : ChunkRequest
            Return Layer_info for this request.

        Raises
        ------
        KeyError
            If the layer is not found.
        """
        layer_id = request.key.layer_key.layer_id

        # Raises KeyError if not found. This should never happen because we
        # add the layer to the layer_map in ChunkLoader.create_request().
        return self.layer_map[layer_id]

    def on_layer_deleted(self, layer):
        """The layer was deleted, delete it from our map.

        Layer
            The layer that was deleted.
        """
        try:
            del self.layer_map[id(layer)]
        except KeyError:
            pass  # We weren't tracking that layer yet.

    def wait_for_all(self):
        """Wait for all in-progress requests to finish."""
        self.delay_queue.flush()

        for future_list in self.futures.values():
            # Result blocks until the future is done or cancelled
            map(lambda x: x.result(), future_list)

    def wait_for_data_id(self, data_id: int) -> None:
        """Wait for the given data to be loaded.

        Parameters
        ----------
        data_id : int
            Wait on chunks for this data_id.
        """
        try:
            future_list = self.futures[data_id]
        except KeyError:
            LOGGER.warning(
                "ChunkLoader.wait: no futures for data_id %d", data_id
            )
            return

        LOGGER.info(
            "ChunkLoader.wait: waiting on %d futures for %d",
            len(future_list),
            data_id,
        )

        # Calling result() will block until the future has finished or was
        # cancelled.
        map(lambda x: x.result(), future_list)
        del self.futures[data_id]


@contextmanager
def synchronous_loading(enabled):
    """Context object to enable or disable async loading.

    with synchronous_loading(True):
        layer = Image(data)
        ... use layer ...
    """
    previous = chunk_loader.force_synchronous
    chunk_loader.force_synchronous = enabled
    yield
    chunk_loader.force_synchronous = previous


def wait_for_async():
    """Wait for all asynchronous loads to finish."""
    chunk_loader.wait_for_all()


"""
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
chunk_loader = ChunkLoader() if octree_config else None

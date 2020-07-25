"""ChunkLoader and the synchronous_loading context object.

There is one global chunk_loader instance to handle async loading for all
Viewer instances that are running. There are two main reasons we just have
one ChunkLoader and not one per Viewer:

1. We size the ChunkCache as a fraction of RAM, so having more than one
   would use too much RAM.

2. We (will) size the thread pool for optimal performance, and having
   multiple pools would result in more workers than we want.

Think of the ChunkLoader as a shared resource like "the filesystem" where
multiple clients can be access it at the same time.
"""
from contextlib import contextmanager
from concurrent import futures
import logging
import threading
import time
from typing import Dict, List, Optional
import weakref

import numpy as np

from ...types import ArrayLike
from ...utils.event import EmitterGroup

from ._cache import ChunkCache
from ._config import async_config
from ._request import ChunkRequest
from ..perf import perf_timer

LOGGER = logging.getLogger("ChunkLoader")


def _chunk_loader_worker(request: ChunkRequest):
    """The worker thread or process that loads the array.

    We have workers because when we call np.asarray() that might lead
    to IO or computation which could take a while. We do not want to
    do that in the GUI thread or the UI will block.

    """
    request.load_chunks()
    return request


def _create_executor(use_processes: bool, num_workers: int):
    """Return the thread or process pool executor.
    """
    executor_class = (
        futures.ProcessPoolExecutor
        if use_processes
        else futures.ThreadPoolExecutor
    )
    return executor_class(max_workers=num_workers)


class ChunkQueueEntry:
    def __init__(self, request, submit_time):
        self.request = request
        self.submit_time = submit_time


class ChunkDelayQueue(threading.Thread):
    """A threaded queue that delays request submission a bit.

    We delay submitting requests so that if you are zipping through slices
    we don't end up starting a bunch of workers for stale slices where no
    one will use the results. We can't cancel workers once they've started,
    but we can trivially cancel requests still in this queue.

    For example if the user is continuously move the slider requests will
    got into this queue and then get immediately canceled as the user
    goes to the next slice. Only when they pause by self.delay_seconds
    will the request actually get submitting the worker start loading.

    Attributes
    ----------
    executor : Union[ThreadPoolExecutor, ProcessPoolExecutor]
        Our thread or process pool executor.
    use_processes : bool
        If True use a process pool for the workers, otherwise use a thread pool.
    num_workers : int
        Create this many worker threads or processes.
    """

    def __init__(self, delay_seconds, submit_func):
        super().__init__(daemon=True)
        self.delay_seconds = delay_seconds
        self.submit_func = submit_func
        self.use_processes: bool = async_config.use_processes
        self.num_workers: int = async_config.num_workers
        self.entries = []
        self.start()

    def add(self, request) -> None:
        """Insert the request into the queue.

        Parameters
        ----------
        request : ChunkRequest
            Insert this request into the queue.
        """
        if self.delay_seconds == 0:
            # Submit with no delay.
            self.submit_func(request)
        else:
            # Add to the queue for a short period of time.
            submit_time = time.time() + self.delay_seconds
            self.entries.append(ChunkQueueEntry(request, submit_time))

    def submit(self, entry, now):
        """Submit the return if its time.

        Parameters
        ----------
        entry : ChunkQueueEntry
            The entry to potentially submit.
        now : int
            Current time in seconds.

        Return
        ------
        bool
            True if the entry was submitted.
        """
        if entry.submit_time < now:
            self.submit_func(entry.request)
            return True
        return False

    def clear(self, data_id):
        """Remove any entires for this data_id.

        Parameters
        ----------
        data_id : int
            Remove entries for this data_id.
        """
        self.entries = [
            x for x in self.entries if x.request.data_id != data_id
        ]

    def run(self):
        """The thread's main method.

        Submit requests after their delay is up.
        """
        while True:
            # Submit all entires which are due
            now = time.time()
            self.entries = [x for x in self.entries if not self.submit(x, now)]
            time.sleep(self.delay_seconds)


class ChunkLoader:
    """Load chunks in workers.

    Calling np.asarray on request.array might block, so we do it in a
    worker thread or process.

    ChunkLoader runs synchronously if ChunkLoader.synchronous is True
    or if the chunk was found in the cache. Otherwise it's asynchronous
    and load_chunk() will return immediately.

    Attributes
    ----------
    synchronous : bool
        If True the ChunkLoader is essentially disabled, loads are done
        immediately and in the GUI thread. If False loads are done
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
    LayerMap = Dict[int, weakref.ReferenceType]

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
        self.executor = _create_executor(use_processes, num_workers)

        # Delay queue prevents us from spamming the worker pool when the
        # user is rapidly scrolling through slices.
        self.delay_queue = ChunkDelayQueue(
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

        # Maps layer_id to weakref of a Layer. ChunkRequest cannot hold a
        # layer reference or even a weakref because with worker processes
        # those cannot be pickled. So ChunkRequest just holds a layer_id
        # that we can map back to a weakref.
        self.layer_map: self.LayerMap = {}

    def create_request(self, layer, indices, chunks: Dict[str, ArrayLike]):
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
        # Set the layer_id to weakref mapping, we can't store the layer
        # reference in the request, so we just store its id.
        self.layer_map[id(layer)] = weakref.ref(layer)

        # Return the new request.
        return ChunkRequest(layer, indices, chunks)

    def _asarray(self, array):
        """Get the array data. We break this out for perfmon timing.
        """
        return np.asarray(array)

    def load_chunk(self, request: ChunkRequest) -> Optional[ChunkRequest]:
        """Process the given ChunkRequest, load its data.

        If ChunkLoader is synchronous or the chunk is in the cache, the
        load is performed immediately in the GUI thread and the satisfied
        request is returned.

        Otherwise an asynchronous load is requested and None is returned.
        The load will be performed in a worker thread and later
        Layer.chunk_loaded() will be called in the GUI thread with the
        satisfied result.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.

        ChunkRequest, optional
            The satisfied ChunkRequest or None indicating an async load.

        Returns
        -------
        Optional[ChunkRequest]
            The ChunkRequest if it was satisfied otherwise None.
        """
        # We do an immediate load in the GUI thread in three cases:
        # 1. We are in synchronous mode.
        # 2. The request contains only ndarrays which are in memory.
        # 3. We are inserting a delay in front of all requests.
        if (
            self.synchronous
            or request.in_memory
            and not self.force_delay_seconds
        ):
            LOGGER.info("[sync] ChunkLoader.load_chunk")
            request.load_chunks_gui()
            return request

        # Check the cache.
        chunks = self.cache.get_chunk(request)

        if chunks is not None:
            LOGGER.info(
                "[async] ChunkLoader._load_async: cache hit %s", request.key
            )
            request.chunks = chunks
            return request

        LOGGER.info(
            "[async] ChunkLoader.load_chunk: cache miss %s", request.key
        )

        # Clear any pending request for this specific data_id. We only
        # support single tile images today, so there can only be one load
        # in progress per data_id.
        with perf_timer("clear_pending"):
            self._clear_pending(request.data_id)

        # Set the extra load sleep time if any.
        request.load_seconds = self.load_seconds

        # Add to the delay queue, it will call our _submit_async() if the
        # delay expires without the request getting cancelled.
        self.delay_queue.add(request)

    def _submit_async(self, request: ChunkRequest) -> None:
        """Initiate an asynchronous load of the given request.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.
        """
        LOGGER.info("[async] ChunkLoader._load_async: %s", request.key)

        future = self.executor.submit(_chunk_loader_worker, request)
        future.add_done_callback(self._done)

        # Add future as in-progress for this data_id.
        self.futures.setdefault(request.data_id, []).append(future)

        # Async load was started, request was not satisfied yet.
        return None

    def _clear_pending(self, data_id: int) -> None:
        """Clear any pending requests for this data_id.

        Parameters
        ----------
        data_id : int
            Clear all requests associated with this data_id.
        """
        # Clear requests that have not even been submitting to the worker pool.
        self.delay_queue.clear(data_id)

        # Get any futures that have been submitted to the pool.
        future_list = self.futures.setdefault(data_id, [])

        # Try to cancel them all, but cancel() will return False if the
        # task already started running.
        num_before = len(future_list)
        future_list[:] = [x for x in future_list if x.cancel()]
        num_after = len(future_list)
        num_cleared = num_before - num_after

        # Delete it entirely if empty
        if num_after == 0:
            del self.futures[data_id]

        if num_before == 0:
            LOGGER.info("[async] ChunkLoader.clear_pending: empty")
        else:
            LOGGER.info(
                "[async] ChunkLoader.clear_pending: %d of %d cleared -> %d remain",
                num_cleared,
                num_before,
                num_after,
            )

    def _done(self, future: futures.Future) -> None:
        """Called when a future finished with success or was cancelled.

        Notes
        -----
        This method may be called in the worker thread. The documentation
        very intentionally does not specify which thread the callback will
        be called in, only that it will be called in the same process. On
        MacOS it seems to always be called in the worker thread with a
        thread pool.
        """
        try:
            request = future.result()
        except futures.CancelledError:
            LOGGER.info("[async] ChunkLoader._done: cancelled")
            return

        LOGGER.info("[async] ChunkLoader._done: %s", request.key)

        # Now that we are back in this process we can add the perf events.
        # If using threads it doesn't matter, but we do it the same way.
        request.add_perf_events()

        # Add chunk to the cache in the worker thread. For now it's safe.
        # Later we might need to arrange for this to be done in the GUI thread
        # if cache access from other threads is not safe.
        self.cache.add_chunk(request)

        # Convert the requests layer_id into a real layer reference.
        layer = self._get_layer_for_request(request)

        # The layer_id was not found or the weakref failed to resolve.
        if layer is None:
            LOGGER.info(
                "[async] ChunkLoader._done: layer not found %d",
                request.layer_id,
            )
            return

        # Signal the chunk was loaded. QtChunkReceiver listens for this and
        # will forward the chunk to the layer in the GUI thread.
        self.events.chunk_loaded(layer=layer, request=request)

    def _get_layer_for_request(self, request: ChunkRequest):
        """Return Layer associated with this request or None.
        """
        layer_id = request.layer_id

        try:
            layer_ref = self.layer_map[layer_id]
        except KeyError:
            LOGGER.warn("[async] ChunkLoader._done: no layer_id %d", layer_id)
            return None

        # Could return None if Layer was deleted.
        return layer_ref()

    def wait(self, data_id: int) -> None:
        """Wait for the given data to be loaded.

        TODO_ASYNC: not using this yet but might be needed?

        Parameters
        ----------
        data_id : int
            Wait on chunks for this data_id.
        """
        try:
            future_list = self.futures[data_id]
        except KeyError:
            LOGGER.warn(
                "[async] ChunkLoader.wait: no futures for data_id %d", data_id
            )
            return

        LOGGER.info(
            "[async] ChunkLoader.wait: waiting on %d futures for %d",
            len(future_list),
            data_id,
        )

        [future.result() for future in future_list]
        del self.futures[data_id]

    def on_layer_deleted(self, layer):
        """TODO_ASYNC: Temporary as a test"""
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

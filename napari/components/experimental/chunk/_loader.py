"""ChunkLoader class.

Loads chunks synchronously, or asynchronously using worker threads or
processes. A chunk could be an OctreeChunk or it could be a pre-Octree
array from the Image class, time-series or multi-scale.
"""
import logging
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

from ....utils.config import octree_config
from ....utils.events import EmitterGroup
from ._cache import ChunkCache
from ._info import LayerInfo, LoadType
from ._pool_group import LoaderPoolGroup
from ._request import ChunkRequest

LOGGER = logging.getLogger("napari.loader")


class ChunkLoader:
    """Loads chunks in worker threads or processes.

    A ChunkLoader contains one or more LoaderPools. Each LoaderPool has
    a thread or process pool.

    Attributes
    ----------
    layer_map : Dict[int, LayerInfo]
        Stores a LayerInfo about each layer we are tracking.
    cache : ChunkCache
        Cache of previously loaded chunks.
    events : EmitterGroup
        We only signal one event: chunk_loaded.
    """

    def __init__(self):
        _setup_logging(octree_config)

        loader_config = octree_config['loader_defaults']

        self.force_synchronous: bool = bool(loader_config['force_synchronous'])
        self.auto_sync_ms = loader_config['auto_sync_ms']
        self.octree_enabled = octree_config['octree']['enabled']

        self.layer_map: Dict[int, LayerInfo] = {}
        self.cache: ChunkCache = ChunkCache()

        self.events = EmitterGroup(
            source=self, auto_connect=True, chunk_loaded=None
        )

        self._loaders = LoaderPoolGroup(octree_config, self._on_done)

    def get_info(self, layer_id: int) -> Optional[LayerInfo]:
        """Get LayerInfo for this layer or None.

        Parameters
        ----------
        layer_id : int
            The the LayerInfo for this layer.

        Returns
        -------
        Optional[LayerInfo]
            The LayerInfo if the layer has one.
        """
        return self.layer_map.get(layer_id)

    def load_request(
        self, request: ChunkRequest
    ) -> Tuple[Optional[ChunkRequest], Optional[Future]]:
        """Load the given request sync or async.

        Parameters
        ----------
        request : ChunkRequest
            Contains one or more arrays to load.

        Returns
        -------
        Tuple[Optional[ChunkRequest], Optional[Future]]
            The ChunkRequest if loaded sync or the Future if loaded async.

        Notes
        -----
        We return a ChunkRequest if the load was performed synchronously,
        otherwise we return a Future meaning an asynchronous load was
        intitiated. When the async load finishes the layer's
        on_chunk_loaded() will be called from the GUI thread.
        """
        self._add_layer_info(request)

        if self._load_synchronously(request):
            return request

        # Check the cache first.
        chunks = self.cache.get_chunks(request)

        if chunks is not None:
            request.chunks = chunks
            return request

        self._loaders.load_async(request)
        return None  # None means load was async.

    def _add_layer_info(self, request: ChunkRequest) -> None:
        """Add a new LayerInfo entry in our layer map.

        Parameters
        ----------
        request : ChunkRequest
            Add a LayerInfo for this request.
        """
        layer_id = request.location.layer_id
        if layer_id not in self.layer_map:
            self.layer_map[layer_id] = LayerInfo(
                request.location.layer_ref, self.auto_sync_ms
            )

    def cancel_requests(
        self, should_cancel: Callable[[ChunkRequest], bool]
    ) -> List[ChunkRequest]:
        """Cancel pending requests based on the given filter.

        Parameters
        ----------
        should_cancel : Callable[[ChunkRequest], bool]
            Cancel the request if this returns True.

        Returns
        -------
        List[ChunkRequests]
            The requests that were cancelled, if any.
        """
        return self._loaders.cancel_requests(should_cancel)

    def _load_synchronously(self, request: ChunkRequest) -> bool:
        """Return True if we loaded the request.

        Attempt to load the request synchronously.

        Parameters
        ----------
        request : ChunkRequest
            The request to load.

        Returns
        -------
        bool
            True if we loaded it.
        """
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

        # TODO_OCTREE: we no longer do auto-sync, this would be need to be
        # implement in a nice way for octree?
        # if info.loads_fast:
        #    return True

        # Finally, load synchronously if it's an ndarray (in memory) otherwise
        # it's Dask or something else and we load async.
        return request.in_memory

    def _on_done(self, request: ChunkRequest) -> None:
        """Called when a future finishes with success or was cancelled.

        Parameters
        ----------
        request : Future
            The future that finished or was cancelled.

        Notes
        -----
        This method MAY be called in a worker thread. The
        concurrent.futures documentation intentionally does not specify
        which thread the future's done callback will be called in, only
        that it will be called in some thread in the current process.
        """
        LOGGER.debug(
            "_done: load=%.3fms elapsed=%.3fms %s",
            request.load_ms,
            request.elapsed_ms,
            request.location,
        )

        # Add chunks to the cache in the worker thread. For now it's safe
        # to do this in the worker. Later we might need to arrange for this
        # to be done in the GUI thread if cache access becomes more
        # complicated.
        self.cache.add_chunks(request)

        # Lookup this request's LayerInfo.
        info = self._get_layer_info(request)

        # Resolve the weakref.
        layer = info.get_layer()

        if layer is None:
            return  # Ignore chunks since layer was deleted.

        info.stats.on_load_finished(request, sync=False)

        # Fire chunk_loaded event  to tell QtChunkReceiver to forward this
        # chunk to its layer in the GUI thread.
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
        layer_id = request.location.layer_id

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

        for future_list in self._futures.values():
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
            future_list = self._futures[data_id]
        except KeyError:
            LOGGER.warning(
                "wait_for_data_id: no futures for data_id=%d", data_id
            )
            return

        LOGGER.info(
            "wait_for_data_id: waiting on %d futures for data_id=%d",
            len(future_list),
            data_id,
        )

        # Calling result() will block until the future has finished or was
        # cancelled.
        map(lambda x: x.result(), future_list)
        del self._futures[data_id]

    def shutdown(self) -> None:
        """When napari is shutting down."""
        self._loaders.shutdown()


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


def _setup_logging(config: dict) -> None:
    """Setup logging.

    Notes
    -----
    It's recommended to use the oldest style of string formatting with
    logging. With f-strings you'd pay the price of formatting the string
    even if the log statement is disabled due to the log level, etc. In our
    case the log will almost always be disabled unless debugging.
    https://docs.python.org/3/howto/logging.html#optimization
    https://blog.pilosus.org/posts/2020/01/24/python-f-strings-in-logging/

    Parameters
    ----------
    config : dict
        The configuration data.
    """
    try:
        log_path = config['loader_defaults']['log_path']
        if log_path is not None:
            _log_to_file("napari.loader", log_path)
    except KeyError:
        pass

    try:
        log_path = config['octree']['log_path']
        if log_path is not None:
            _log_to_file("napari.octree", log_path)
    except KeyError:
        pass


def _log_to_file(name: str, path: str) -> None:
    """Log "name" messages to the given file path.

    Parameters
    ----------
    path : str
        Log to this file path.
    """
    log_format = "%(levelname)s - %(name)s - %(message)s"
    logger = logging.getLogger(name)
    fh = logging.FileHandler(path)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


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

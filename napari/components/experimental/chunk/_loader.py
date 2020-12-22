"""ChunkLoader class.

Loads chunks synchronously or asynchronously using worker threads or
processes. A chunk could be an OctreeChunk or it could be a pre-Octree
array from a single or multi-scale image.
"""
import logging
from concurrent.futures import CancelledError, Future
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

from ....types import ArrayLike
from ....utils.config import octree_config
from ....utils.events import EmitterGroup
from ._cache import ChunkCache
from ._info import LayerInfo, LayerRef, LoadType
from ._pool import LoaderPool
from ._request import ChunkKey, ChunkRequest

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

        loader_config = octree_config['loader']

        self.force_synchronous: bool = bool(loader_config['force_synchronous'])
        self.auto_sync_ms = loader_config['auto_sync_ms']
        self.octree_enabled = octree_config['octree']['enabled']

        self.layer_map: Dict[int, LayerInfo] = {}
        self.cache: ChunkCache = ChunkCache()

        self.events = EmitterGroup(
            source=self, auto_connect=True, chunk_loaded=None
        )

        self._loader = LoaderPool(loader_config, self._done)

    def get_info(self, layer_id: int) -> Optional[LayerInfo]:
        """Get LayerInfo for this layer or None."""
        return self.layer_map.get(layer_id)

    def create_request(
        self, layer_ref: LayerRef, key: ChunkKey, chunks: Dict[str, ArrayLike]
    ) -> ChunkRequest:
        """Create a ChunkRequest for submission to load_chunk.

        TODO_OCTREE: We have this method mainly so we can create an
        entry in self.layer_map. But it seems like it would be simpler
        if users of ChunkLoader could just create a ChunkRequest on
        their own. And we create the layer_map entry on submission?
        These seems kind of historical?

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
            self.layer_map[layer_id] = LayerInfo(layer_ref, self.auto_sync_ms)

        # Return the new request.
        return ChunkRequest(key, chunks)

    def load_chunk(
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
        if self._load_synchronously(request):
            return request

        # Check the cache first.
        chunks = self.cache.get_chunks(request)

        if chunks is not None:
            request.chunks = chunks
            return request

        if not self.octree_enabled:
            # Pre-octree we can clear pendint requests from any other data_id,
            # generally from other slices besides this one.
            self._loader.clear_pending(request.data_id)

        self._loader.load_async(request)
        return None  # None means load was async.

    def _load_synchronously(self, request: ChunkRequest) -> bool:
        """Return True if we loaded the request.

        Attempt to load the request synchronously.

        Parameters
        ----------
        request : ChunkRequest
            The request to load.

        Return
        ------
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

        # TODO_OCTREE: disable this in a nice way for octree. It made sense
        # for single-scale time series, but not for octree.
        # if info.loads_fast:
        #    return True

        # Finally, load synchronously if it's an ndarray (in memory) otherwise
        # it's Dask or something else and we load async.
        return request.in_memory

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
            # never block. But we can see if it finished or was
            # cancelled. Although we don't care right now.
            return future.result()
        except CancelledError:
            return None

    def _done(self, future: Future) -> None:
        """Called when a future finishes with success or was cancelled.

        Parameters
        ----------
        future : Future
            The future that finished or was cancelled.

        Notes
        -----
        This method MAY be called in a worker thread. The
        concurrent.futures documentation very intentionally does not
        specify which thread the future's done callback will be called in,
        only that it will be called in some thread in the current process.
        """
        try:
            request = self._get_request(future)
        except ValueError:
            return  # Pool not running? App exit in progress?

        if request is None:
            return  # Future was cancelled, nothing to do.

        LOGGER.debug(
            "_done: load=%.3fms elapsed=%.3fms location=%s",
            request.load_ms,
            request.elapsed_ms,
            request.key.location,
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

    String Formatting
    -----------------
    It's recommended to use the oldest style of string formatting with
    logging. With f-strings you'd pay the price of formatting the string
    even if the log statement is disabled due to the log level, etc. In our
    case the log will almost always be disabled unless debugging.
    https://docs.python.org/3/howto/logging.html#optimization
    https://blog.pilosus.org/posts/2020/01/24/python-f-strings-in-logging/

    Parameters
    ----------
    octree_config : dict
        The configuration data.
    """
    try:
        log_path = config['loader']['log_path']
        if log_path is not None:
            _log_to_file("napari.loader", log_path)
    except KeyError:
        pass

    try:
        log_path = config['loader']['log_path']
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

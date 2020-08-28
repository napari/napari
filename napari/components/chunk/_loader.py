"""ChunkLoader: synchronous or asynchronous chunk loading
"""
import logging
import os
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from typing import Dict, List, Optional

from ...types import ArrayLike
from ...utils.events import EmitterGroup
from ._info import LayerInfo
from ._request import ChunkKey, ChunkRequest

LOGGER = logging.getLogger("napari.async")


def _is_enabled(env_var) -> bool:
    """Return True if env_var is defined and not zero."""
    return os.getenv(env_var, "0") != "0"


def _chunk_loader_worker(request: ChunkRequest) -> ChunkRequest:
    """This is the worker thread that loads the array.

    We call np.asarray() in a worker because it might lead to IO or
    computation which would block the GUI thread.

    Parameters
    ----------
    request : ChunkRequest
        The request to load.
    """
    request.load_chunks()  # loads all chunks in the request
    return request


class ChunkLoader:
    """Loads chunks synchronously or asynchronously in worker threads.

    We cannot call np.asarray() in the GUI thread because it might block on
    IO or a computation. So the ChunkLoader calls np.asarray() in a worker
    thread if doing async loading.

    Attributes
    ----------
    synchronous : bool
        If True all requests are loaded synchronously.
    executor : ThreadPoolExecutor
        Our thread pool executor.
    futures : Dict[int, List[Future]]
        In progress futures for each layer (data_id).
    events : EmitterGroup
        We only signal one event: chunk_loaded.
    layer_map : Dict[int, LayerInfo]
        Stores a LayerInfo about each layer we are tracking.
    """

    def __init__(self):
        self.synchronous = not _is_enabled("NAPARI_ASYNC")
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.futures: Dict[int, List[Future]] = {}
        self.events = EmitterGroup(
            source=self, auto_connect=True, chunk_loaded=None
        )
        self.layer_map: Dict[int, LayerInfo] = {}

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
        key : ChunkKey
            The key for the request.
        chunks : Dict[str, ArrayLike]
            The arrays we want to load.
        """
        layer_id = key.layer_id

        # Add a LayerInfo if we don't already have one.
        if layer_id not in self.layer_map:
            self.layer_map[layer_id] = LayerInfo(layer)

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

        # Clear any pending requests for this specific data_id.
        self._clear_pending(request.key.data_id)

        # Sumbit the request asynchronously.
        self._submit_async(request)

    def _load_synchronously(self, request: ChunkRequest) -> bool:
        """Return True if we loaded the request synchronously."""

        # If async is enabled, loaded non-ndarray request async.
        if not self.synchronous and not request.in_memory:
            return False  # load async

        # Load synchronously.
        request.load_chunks()
        return True

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
        LOGGER.debug("ChunkLoader._load_async: %s", request.key)

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
        LOGGER.debug("ChunkLoader._clear_pending %d", data_id)

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
        This method may be called in the worker thread. The
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

        # Lookup this Request's LayerInfo.
        info = self._get_layer_info(request)

        # Resolve the weakref.
        layer = info.get_layer()

        if layer is None:
            return  # Ignore chunks since layer was deleted.

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
        layer_id = request.key.layer_id

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


# Global instance
chunk_loader = ChunkLoader()

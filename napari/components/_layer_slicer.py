from __future__ import annotations

import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import RLock
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import EmitterGroup, Event

logger = logging.getLogger("napari.components._layer_slicer")


# Layers that can be asynchronously sliced must be able to make
# a slice request that can be called and will produce a slice
# response. The request and response types will vary per layer
# type, which means that the values of the dictionary result of
# ``_slice_layers`` cannot be fixed to a single type.

_SliceResponse = TypeVar('_SliceResponse')
_SliceRequest = Callable[[], _SliceResponse]


@runtime_checkable
class _AsyncSliceable(Protocol[_SliceResponse]):
    def _make_slice_request(self, dims: Dims) -> _SliceRequest[_SliceResponse]:
        ...

    def _update_slice_response(self, response: _SliceResponse) -> None:
        ...


class _LayerSlicer:
    """
    High level class to control the creation of a slice (via a slice request),
    submit it (synchronously or asynchronously) to a thread pool, and emit the
    results when complete.

    Events
    ------
    ready
        emitted after slicing is done with a dict value that maps from layer
        to slice response. Note that this may be emitted on the main or
        a non-main thread. If usage of this event relies on something happening
        on the main thread, actions should be taken to ensure that the callback
        is also executed on the main thread (e.g. by decorating the callback
        with `@ensure_main_thread`).
    """

    def __init__(self):
        """
        Attributes
        ----------
        _executor : concurrent.futures.ThreadPoolExecutor
            manager for the slicing threading
        _force_sync: bool
            if true, forces slicing to execute synchronously
        _layers_to_task : dict
            task storage for cancellation logic
        _lock_layers_to_task : threading.RLock
            lock to guard against changes to `_layers_to_task` when finding,
            adding, or removing tasks.
        """
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._force_sync = True
        self._layers_to_task: Dict[Tuple[Layer], Future] = {}
        self._lock_layers_to_task = RLock()

    @contextmanager
    def force_sync(self):
        """Context manager to temporarily force slicing to be synchronous.
        This should only be used from the main thread.

        >>> layer_slicer = _LayerSlicer()
        >>> layer = Image(...)  # an async-ready layer
        >>> with layer_slice.force_sync():
        >>>     layer_slicer.slice_layers_async(layers=[layer], dims=Dims())
        """
        prev = self._force_sync
        self._force_sync = True
        try:
            yield None
        finally:
            self._force_sync = prev

    def wait_until_idle(self, timeout: Optional[float] = None) -> None:
        """Wait for all slicing tasks to complete before returning.

        Attributes
        ----------
        timeout: float or None
            (Optional) time in seconds to wait before raising TimeoutError. If set as None,
            there is no limit to the wait time. Defaults to None

        Raises
        ------
        TimeoutError: when the timeout limit has been exceeded and the task is
            not yet complete
        """
        futures = self._layers_to_task.values()
        _, not_done_futures = wait(futures, timeout=timeout)

        if len(not_done_futures) > 0:
            raise TimeoutError(
                f'Slicing {len(not_done_futures)} tasks did not complete within timeout ({timeout}s).'
            )

    def slice_layers_async(
        self, layers: Iterable[Layer], dims: Dims
    ) -> Optional[Future[dict]]:
        """This should only be called from the main thread.

        Creates a new task and adds it to the _layers_to_task dict. Cancels
        all tasks currently pending for that layer tuple.

        Submitting multiple layers at one generates multiple requests (stored in a dict),
        but only ONE task.

        If multiple layers are sliced, any task that contains only one of those
        layers can safely be cancelled. If a single layer is sliced, it will
        wait for any existing tasks that include that layer AND another layer,
        In other words, it will only cancel if the new task will replace the
        slices of all the layers in the pending task.

        TODO: consider renaming this slice_layers, or maybe just slice, or run;
        we don't know if slicing will be async or not.
        """
        # Cancel any tasks that are slicing a subset of the layers
        # being sliced now. This allows us to slice arbitrary sets of
        # layers with some sensible and not too complex cancellation
        # policy.
        if existing_task := self._find_existing_task(layers):
            logger.debug('Cancelling task for %s', layers)
            existing_task.cancel()

        # Not all layer types will initially be asynchronously sliceable.
        # The following logic gives us a way to handle those in the short
        # term as we develop, and also in the long term if there are cases
        # when we want to perform sync slicing anyway.
        requests = {}
        sync_layers = []
        for layer in layers:
            if isinstance(layer, _AsyncSliceable) and not self._force_sync:
                logger.debug('Async slicing: %s', layer)
                requests[layer] = layer._make_slice_request(dims)
            else:
                sync_layers.append(layer)

        # create task for all requests
        task = None
        if len(requests) > 0:
            task = self._executor.submit(self._slice_layers, requests)

            # store task for cancellation logic
            # this is purposefully done before adding the done callback to ensure
            # that the task is added before the done callback can be executed
            with self._lock_layers_to_task:
                self._layers_to_task[tuple(requests)] = task

            task.add_done_callback(self._on_slice_done)

        # slice the sync layers after async submission so that async
        # tasks can potentially run concurrently
        for layer in sync_layers:
            logger.debug('Sync slicing: %s', layer)
            layer._slice_dims(dims.point, dims.ndisplay, dims.order)

        return task

    def shutdown(self) -> None:
        """Should be called from the main thread when this is no longer needed."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _slice_layers(self, requests: Dict) -> Dict:
        """
        Iterates through a dictionary of request objects and call the slice
        on each individual layer. Can be called from the main or slicing thread.

        Attributes
        ----------
        requests: dict[Layer, SliceRequest]
            Dictionary of request objects to be used for constructing the slice

        Returns
        -------
        dict[Layer, SliceResponse]: which contains the results of the slice
        """
        return {layer: request() for layer, request in requests.items()}

    def _on_slice_done(self, task: Future[Dict]) -> None:
        """
        This is the "done_callback" which is added to each task.
        Can be called from the main or slicing thread.
        """
        if not self._try_to_remove_task(task):
            logger.debug('Task not found')
            return

        if task.cancelled():
            logger.debug('Cancelled task')
            return
        result = task.result()
        self.events.ready(Event('ready', value=result))

    def _try_to_remove_task(self, task: Future[Dict]) -> bool:
        """
        Attempt to remove task, return false if task not found, return true
        if task is found and removed from layers_to_task dict.

        This function provides a lock to ensure that the layers_to_task dict
        is unmodified during this process.
        """
        with self._lock_layers_to_task:
            for k_layers, v_task in self._layers_to_task.items():
                if v_task == task:
                    del self._layers_to_task[k_layers]
                    return True
        return False

    def _find_existing_task(
        self, layers: Iterable[Layer]
    ) -> Optional[Future[Dict]]:
        """Find the task associated with a list of layers. Returns the first
        task found for which the layers of the task are a subset of the input
        layers.

        This function provides a lock to ensure that the layers_to_task dict
        is unmodified during this process.
        """
        with self._lock_layers_to_task:
            layer_set = set(layers)
            for task_layers, task in self._layers_to_task.items():
                if set(task_layers).issubset(layer_set):
                    logger.debug(f'Found existing task for {task_layers}')
                    return task
        return None

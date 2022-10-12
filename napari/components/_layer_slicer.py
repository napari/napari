from __future__ import annotations

import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import RLock
from typing import Dict, Iterable, Optional, Tuple

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import EmitterGroup, Event

logger = logging.getLogger("napari.components._layer_slicer")


class _LayerSlicer:
    """
    High level class to control the creation of a slice (via a slice request),
    submit it (synchronously or asynchronously) to a thread pool, and emit the
    results when complete.

    Attributes
    ----------
    events : napari.utils.events.event.EmitterGroup
        vispy event emitter
    _executor : concurrent.futures.ThreadPoolExecutor
        manager for the slicing threading
    _layers_to_task : dict
        task storage for cancellation logic
    lock_layers_to_task : threading.RLock
        lock to guard against changes to `_layers_to_task` when finding or
        removing tasks.

    Methods
    -------
    slice_layers_async:
        creates the slice and submits it to be executes
    shutdown:
        shutdown the executor on this class. Should only be called when the
        class is no longer being used
    _slice_layers:
        convert request objects to slices
    _on_slice_done:
        callback executed when slicing future is complete
    _try_to_remove_task:
        utility function to remove a task from `_layers_to_task`
    _find_existing_task:
        utility function to find a task in `_layers_to_task`
    """

    def __init__(self):
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._layers_to_task: Dict[Tuple[Layer], Future] = {}
        self.lock_layers_to_task = RLock()

    def slice_layers_async(
        self, layers: Iterable[Layer], dims: Dims
    ) -> Future[dict]:
        """This should only be called from the main thread.

        Creates a new task and adds it to the _layers_to_task dict. Cancels
        all tasks currently pending for that layer tuple.

        Submitting multiple layers at one generates multiple requests (stored in a dict),
        but only ONE task.

        A slice submitted with multiple layers will only cancel if the same tuple of layers
        is resubmitted. If a second slice is requested with only one layer from the first
        slice, it will not be cancelled.
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
        for layer in layers:
            if layer._is_async():
                requests[layer] = layer._make_slice_request(dims)
            else:
                layer._slice_dims(dims.point, dims.ndisplay, dims.order)
        # create task for slicing of each request/layer
        task = self._executor.submit(self._slice_layers, requests)

        # store task for cancellation logic
        self._layers_to_task[tuple(requests.keys())] = task

        task.add_done_callback(self._on_slice_done)

        return task

    def shutdown(self) -> None:
        """This should be called from the main thread when this is no longer needed."""
        self._executor.shutdown()

    def _slice_layers(self, requests: Dict) -> Dict:
        """This can be called from the main or slicing thread.
        Iterates through a dictionary of request objects and call the slice
        on each individual layer."""
        return {
            layer: layer._get_slice(request)
            for layer, request in requests.items()
        }

    def _on_slice_done(self, task: Future[Dict]) -> None:
        """This can be called from the main or slicing thread.
        Release the thread.
        This is the "done_callback" which is added to each task.
        """
        if not self._try_to_remove_task(task):
            logger.debug('Task not found')
            return

        if task.cancelled():
            logger.debug('Cancelled task')
            return
        result = task.result()
        self.events.ready(Event('ready', value=result))

    def _try_to_remove_task(self, task) -> bool:
        """Attempt to remove task, return false if task not found, return true
        if task removed from layers_to_task dict"""
        with self.lock_layers_to_task:
            layers = None
            for k_layers, v_task in self._layers_to_task.items():
                if v_task == task:
                    layers = k_layers
                    break

            if not layers:
                return False
            del self._layers_to_task[layers]
        return True

    def _find_existing_task(self, layers) -> Optional[Future[Dict]]:
        with self.lock_layers_to_task:
            layer_set = set(layers)
            for task_layers, task in self._layers_to_task.items():
                if set(task_layers).issubset(layer_set):
                    logger.debug(f'Found existing task for {task_layers}')
                    return task

        return None

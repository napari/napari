import logging
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Dict, Iterable, Tuple
from threading import Lock
import weakref

from napari.components import Dims
from napari.layers import Layer
from napari.utils.events.event import EmitterGroup, Event

LOGGER = logging.getLogger("napari.components._layer_slicer")


class _LayerSlicer:
    def __init__(self):
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._layers_to_task: Dict[Tuple[Layer], Future] = {}
        self.lock = Lock()

    def slice_layers_async(
        self, layers: Iterable[Layer], dims: Dims
    ) -> Future[dict]:
        """This should only be called from the main thread.

        Creates a new task and adds it to the _layers_to_task dict. Cancels
        all tasks currently pending for that layer.
        """
        # lock to protect against mutations of `task_to_layer` in `_on_slice_done`
        with self.lock:
            # Cancel any tasks that are slicing a subset of the layers
            # being sliced now. This allows us to slice arbitrary sets of
            # layers with some sensible and not too complex cancellation
            # policy.
            layer_set = set(layers)
            for task_layers, task in self._layers_to_task.items():
                if set(task_layers).issubset(layer_set):
                    LOGGER.debug('Cancelling task for %s', task_layers)
                    task.cancel()

            # Not all layer types will initially be asynchronously sliceable.
            # The following logic gives us a way to handle those in the short
            # term as we develop, and also in the long term if there are cases
            # when we want to perform sync slicing anyway.
            requests = {}
            for layer in layers:
                # TODO: temmporary to prove a point
                # assert isinstance(layer, weakref.ReferenceType) # this fails
                if layer._is_async():
                    requests[layer] = layer._make_slice_request(dims)
                else:
                    layer._slice_dims(dims.point, dims.ndisplay, dims.order)
            # create task for slicing of each request/layer
            task = self._executor.submit(self._slice_layers, requests)

            # construct dict of layers to layer task
            self._layers_to_task[tuple(requests.keys())] = task

        # once everything is complete, release the block
        task.add_done_callback(self._on_slice_done)

        return task

    def shutdown(self) -> None:
        """This should be called from the main thread when this is no longer needed."""
        # TODO: kcp: doesn't this make this class instance null? Maybe we need a context manager to handle all instances off this class?
        self._executor.shutdown()

    def _slice_layers(self, requests: Dict) -> Dict:
        """This can be called from the main or slicing thread.
        Iterates throught a dictionary of request objects and call the slice
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
        # lock to protect against mutations of `task_to_layer` in `slice_layers_async``
        with self.lock:
            # TODO: remove task from _layers_to_task, guarding access to dict with a lock.
            # TODO: is it ever possible to get duplicate tasks? if so, the inverse dict can't handle it
            # TODO: kcp: I'm not happy with this ordering, it feels like we shouldn't remove it from the dict before 
            #       we actually call result, though I guess it doens't hurt anything to do so
            task_to_layers = {v: k for k, v in self._layers_to_task.items()}
            layers = task_to_layers.get(task, None)

            if not layers:
                LOGGER.debug('Task not found')
                return

            # remove task
            del self._layers_to_task[layers]

            if task.cancelled():
                LOGGER.debug('Cancelled task')
                return
            result = task.result()
            self.events.ready(Event('ready', value=result))

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Iterable, Optional

from napari.layers import Image, Layer, Points
from napari.utils.events.event import EmitterGroup, Event

from . import Dims

# Type annotations cause a cyclic dependency, so omit for now.
_ViewerSliceRequest = dict  # [Layer, _LayerSliceRequest]
_ViewerSliceResponse = dict  # [Layer, _LayerSliceResponse]


class _LayerSlicer:
    def __init__(self):
        self.events = EmitterGroup(source=self, ready=Event)
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._task: Optional[Future[_ViewerSliceResponse]] = None

    def slice_layers_async(
        self, layers: Iterable[Layer], dims: Dims
    ) -> Future[_ViewerSliceResponse]:
        """This should only be called from the main thread."""
        if self._task is not None:
            self._task.cancel()
        # Not all layer types will initially be asynchronously sliceable.
        # The following logic gives us a way to handle those in the short
        # term as we develop, and also in the long term if there are cases
        # when we want to perform sync slicing anyway.
        requests = {}
        for layer in layers:
            if _is_async(layer):
                requests[layer] = layer._make_slice_request(dims)
            else:
                layer._slice_dims(dims.point, dims.ndisplay, dims.order)
        self._task = self._executor.submit(self._slice_layers, requests)
        self._task.add_done_callback(self._on_slice_done)
        return self._task

    def _slice_layers(
        self, requests: _ViewerSliceRequest
    ) -> _ViewerSliceResponse:
        """This can be called from the main or slicing thread."""
        return {
            layer: layer._get_slice(request)
            for layer, request in requests.items()
        }

    def _on_slice_done(self, task: Future[_ViewerSliceResponse]) -> None:
        """This can be called from the main or slicing thread."""
        if task.cancelled():
            return
        result = task.result()
        self.events.ready(Event('ready', value=result))


def _is_async(layer: Layer) -> bool:
    return isinstance(layer, (Image, Points))

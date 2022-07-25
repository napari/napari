from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Iterable, Optional

from psygnal import Signal

from napari.layers.base.base import Layer

from . import Dims

_ViewerSliceRequest = dict  # [Layer, _LayerSliceRequest]
_ViewerSliceResponse = dict  # [Layer, _LayerSliceResponse]


class _LayerSlicer:

    ready = Signal(_ViewerSliceResponse)

    def __init__(self):
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._task: Optional[Future[_ViewerSliceResponse]] = None

    def slice_layers_async(self, layers: Iterable[Layer], dims: Dims) -> None:
        """This should only be called from the main thread."""
        if self._task is not None:
            self._task.cancel()
        requests = {layer: layer._make_slice_request(dims) for layer in layers}
        self._task = self._executor.submit(self.slice_layers, requests)
        self._task.add_done_callback(self._on_slice_done)

    def slice_layers(
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
        self.ready.emit(result)

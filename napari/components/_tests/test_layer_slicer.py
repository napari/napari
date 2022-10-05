import time
from collections import deque
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, Callable, Deque, Iterable, Union

import numpy as np
import pytest

from napari.components import Dims
from napari.components._layer_slicer import (
    Task,
    _LayerSlicer,
    _SliceRequest,
    _SliceResponse,
)
from napari.layers._data_protocols import LayerDataProtocol

# pytest napari/components/_tests/test_layer_slicer.py -svv


# @pytest.fixture()
# def async_layer():
#     shape = (10, 15)
#     np.random.seed(0)
#     data = np.random.random(shape)
#     layer = AsyncImage(data)
#     return layer


class _TestLayer:
    """Temporary extension of Image Layer to use as async slicing is built out.
    These methods will eventually become part of the Base Layer or individual
    layers.
    """

    def __init__(self, data: LayerDataProtocol):
        self._data = data
        self.lock = RLock()

    def _make_slice_request(self, dims) -> _SliceRequest:
        return _SliceRequest(data=self._data, index=dims.current_step[0])

    def _get_slice(self, request: _SliceRequest) -> _SliceResponse:
        with self.lock:
            return self._data[request.index]

    def _is_async(self):
        return True


class TestExecutor(Executor):
    def __init__(self):
        self._tasks: Deque[Task] = deque()

    def submit(self, fn, *args, **kwargs) -> Future:
        task = Task(partial(fn, *args, **kwargs))
        self._tasks.append(task)
        return task.future

    def run_specific(self, future: Future) -> None:
        for task in self._tasks:
            if task.future is future:
                task.run()
        raise ValueError('future not found')

    def run_oldest(self):
        task = self._tasks.popleft()
        task.run()

    def run_newest(self):
        task = self._tasks.pop()
        task.run()

    def run_all(self):
        while len(self._tasks) > 0:
            self.run_oldest()

    def map(
        self,
        func,
        *iterables,
        timeout: Union[int, float] = None,
        chunksize: int = 1,
    ) -> Iterable[Future]:
        raise NotImplementedError()

    def shutdown(self, wait=True, *, cancel_futures=False) -> None:
        self._tasks.clear()


@pytest.fixture()
def async_layer():
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = _TestLayer(data)
    layer._executor = TestExecutor()
    return layer


@pytest.fixture()
def viewer_slice_request(layer_slice_request, async_layer):
    """[Layer, _SliceRequest]"""

    requests = {
        async_layer: layer_slice_request,
    }
    return requests


@pytest.fixture()
def viewer_slice_response(layer_slice_response, async_layer):
    """[Layer, _SliceResponse]
    slice response should be a future
    """
    responses = {
        async_layer: layer_slice_response,
    }
    return responses


@pytest.fixture()
def layer_slice_request():
    return _SliceRequest(
        data=None,
        data_to_world=None,
        ndim=None,
        ndisplay=None,
        point=None,
        dims_order=None,
        dims_displayed=None,
        dims_not_displayed=None,
        multiscale=None,
        corner_pixels=None,
        round_index=None,
    )


@pytest.fixture()
def layer_slice_response(layer_slice_request):
    return _SliceResponse(
        request=layer_slice_request,
        data=None,
        data_to_world=None,
    )


def test_slice_layers_async(async_layer):
    layer_slicer = _LayerSlicer()
    dims = Dims()
    assert dims.ndim == 2
    slice_reponse = layer_slicer.slice_layers_async(
        layers=[async_layer],
        dims=dims,
    )
    assert slice_reponse


# def test_slice_layers(viewer_slice_request):
#     """requires
#     Layer._get_slice
#     """
#     layer_slicer = _LayerSlicer()
#     slice_reponse = layer_slicer._slice_layers(
#         requests=viewer_slice_request,
#     )
#     assert isinstance(slice_reponse, dict)


# def test_on_slice_done(layer_slice_response):
#     """TODO to test this properly, it needs to be done at a higher level to
#     check result on the Layer."""

#     # test no errors are raised for a simple submit
#     layer_slicer = _LayerSlicer()
#     with layer_slicer._executor as executor:
#         task = executor.submit(tuple, (1, 2))
#         response = layer_slicer._on_slice_done(
#             task=task,
#         )
#         assert response is None
#         assert task.done()

#     # test cancellation of task
#     layer_slicer = _LayerSlicer()
#     with layer_slicer._executor as executor:
#         task = executor.submit(time.sleep, 0.1)
#         task.cancel()
#         response = layer_slicer._on_slice_done(
#             task=task,
#         )
#         assert response is None
#         assert task.done()


def test_executor():
    layer_slicer = _LayerSlicer()
    with layer_slicer._executor as executor:
        task1 = executor.submit(time.sleep, 0.1)
        task2 = executor.submit(time.sleep, 0.2)
        task1.result()
    assert not task1.running()
    assert not task2.running()

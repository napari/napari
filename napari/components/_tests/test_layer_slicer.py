from dataclasses import dataclass
from threading import RLock, current_thread, main_thread
from typing import Tuple, Union

import numpy as np
import pytest

from napari.components import Dims
from napari.components._layer_slicer import _LayerSlicer
from napari.layers import Image
from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.types import DTypeLike

"""
Cases to consider
- single + multiple layers that supports async (all of layers do support async)
- single + multiple layers that don't support async (all of layers do not support async)
- mix of layers that do and don't support async

Behaviors we want to test:
scheduling logic of the slicer (and not correctness of the slice response value)

- for layers that support async, the slice task should not be run on the main
  thread (we don't want to block the calling)
- for layers that do not support async, slicing should always be done once
  the method returns
- slice requests should be run on the main thread
- pending tasks are cancelled (at least when the new task will slice all
  layers for the pending task)

The fake request, response, and layers exist to give structure against which to
test (class instances and attributes) and to remain isolated from the existing
codebase. They represent what will become real classes in the codebase which
have additional methods and properties that don't currently exist.

Run all tests with:
pytest napari/components/_tests/test_layer_slicer.py -svv
"""


@dataclass(frozen=True)
class FakeSliceRequest:
    id: int = -1


@dataclass(frozen=True)
class FakeSliceResponse:
    id: int = -1


class FakeAsyncLayer:
    def __init__(self):
        self.slice_count = 0
        self.lock = RLock()

    def _make_slice_request(self, dims) -> FakeSliceRequest:
        assert current_thread() == main_thread()
        self.slice_count += 1
        request = FakeSliceRequest(id=self.slice_count)
        return request

    def _get_slice(self, request: FakeSliceRequest) -> FakeSliceResponse:
        assert current_thread() != main_thread()
        with self.lock:
            return FakeSliceResponse(id=request.id)

    def _is_async(self) -> bool:
        return True


class FakeSyncLayer:
    def __init__(self):
        self.slice_count: int = 0

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1

    def _is_async(self) -> bool:
        return False


class LockableData:
    """A wrapper for napari layer data that blocks read-access with a lock.

    This is useful when testing async slicing with real napari layers because
    it allows us to control when slicing tasks complete.
    """

    def __init__(self, data: LayerDataProtocol):
        self.data = data
        self.lock = RLock()

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        with self.lock:
            return self.data[key]


@pytest.fixture()
def layer_slicer():
    layer_slicer = _LayerSlicer()
    yield layer_slicer
    layer_slicer.shutdown()


def test_slice_layers_async_with_one_async_layer_no_block(layer_slicer):
    layer = FakeAsyncLayer()

    future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    assert future.result()[layer].id == 1


def test_slice_layers_async_with_multiple_async_layer_no_block(layer_slicer):
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()

    future = layer_slicer.slice_layers_async(
        layers=[layer1, layer2], dims=Dims()
    )

    assert future.result()[layer1].id == 1
    assert future.result()[layer2].id == 1


def test_slice_layers_async_emits_ready_event_when_done(layer_slicer):
    layer = FakeAsyncLayer()
    event_result = None

    def on_done(event):
        nonlocal event_result
        event_result = event.value

    layer_slicer.events.ready.connect(on_done)

    future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())
    actual_result = future.result()

    assert actual_result is event_result


def test_slice_layers_async_with_one_sync_layer(layer_slicer):
    layer = FakeSyncLayer()
    assert layer.slice_count == 0

    future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert future.result() == {}


def test_slice_layers_async_with_multiple_sync_layer(layer_slicer):
    layer1 = FakeSyncLayer()
    layer2 = FakeSyncLayer()
    assert layer1.slice_count == 0
    assert layer2.slice_count == 0

    future = layer_slicer.slice_layers_async(
        layers=[layer1, layer2], dims=Dims()
    )

    assert layer1.slice_count == 1
    assert layer2.slice_count == 1
    assert not future.result()


def test_slice_layers_async_with_mixed_layers(layer_slicer):
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    assert layer1.slice_count == 0
    assert layer2.slice_count == 0

    future = layer_slicer.slice_layers_async(
        layers=[layer1, layer2], dims=Dims()
    )

    assert layer1.slice_count == 1
    assert layer2.slice_count == 1
    assert future.result()[layer1].id == 1
    assert layer2 not in future.result()


def test_slice_layers_async_lock_blocking(layer_slicer):
    dims = Dims()
    layer = FakeAsyncLayer()

    assert layer.slice_count == 0
    with layer.lock:
        blocked = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        assert not blocked.done()

    assert blocked.result()[layer].id == 1


def test_slice_layers_async_multiple_calls_cancels_pending(layer_slicer):
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        blocked = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        pending = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        assert not pending.running()
        layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        assert not blocked.done()

    assert pending.cancelled()


def test_slice_layers_mixed_allows_sync_to_run(layer_slicer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    with layer1.lock:
        blocked = layer_slicer.slice_layers_async(layers=[layer1], dims=dims)
        layer_slicer.slice_layers_async(layers=[layer2], dims=dims)
        assert layer2.slice_count == 1
        assert not blocked.done()

    assert blocked.result()[layer1].id == 1


def test_slice_layers_mixed_allows_sync_to_run_one_slicer_call(layer_slicer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    with layer1.lock:
        blocked = layer_slicer.slice_layers_async(
            layers=[layer1, layer2], dims=dims
        )

        assert layer2.slice_count == 1
        assert not blocked.done()

    assert blocked.result()[layer1].id == 1


def test_slice_layers_async_with_multiple_async_layer_with_all_locked(
    layer_slicer,
):
    """ensure that if only all layers are locked, none continue"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()

    with layer1.lock, layer2.lock:
        blocked = layer_slicer.slice_layers_async(
            layers=[layer1, layer2], dims=dims
        )
        assert not blocked.done()

    assert blocked.result()[layer1].id == 1
    assert blocked.result()[layer2].id == 1


def test_slice_layers_async_task_to_layers_lock(layer_slicer):
    """ensure that if only one layer has a lock, the non-locked layer
    can continue"""
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        task = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        assert task in layer_slicer._layers_to_task.values()

    assert task.result()[layer].id == 1
    assert task not in layer_slicer._layers_to_task


def test_slice_layers_async_with_one_2d_image(layer_slicer):
    np.random.seed(0)
    data = np.random.rand(8, 7)
    lockable_data = LockableData(data)
    layer = Image(data=lockable_data, multiscale=False)

    with lockable_data.lock:
        future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())
        assert not future.done()

    layer_result = future.result()[layer]
    np.testing.assert_equal(layer_result.data, data)


def test_slice_layers_async_with_one_3d_image(layer_slicer):
    np.random.seed(0)
    data = np.random.rand(8, 7, 6)
    lockable_data = LockableData(data)
    layer = Image(data=lockable_data, multiscale=False)
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 8, 1), (0, 7, 1), (0, 6, 1)),
        current_step=(2, 0, 0),
    )

    with lockable_data.lock:
        future = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        assert not future.done()

    layer_result = future.result()[layer]
    np.testing.assert_equal(layer_result.data, data[2, :, :])

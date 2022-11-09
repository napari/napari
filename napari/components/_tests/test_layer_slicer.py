from concurrent.futures import Future, wait
from dataclasses import dataclass
from threading import RLock, current_thread, main_thread

import pytest

from napari.components import Dims
from napari.components._layer_slicer import _LayerSlicer

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

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1


class FakeSyncLayer:
    def __init__(self):
        self.slice_count: int = 0

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1

    def _is_async(self) -> bool:
        return False


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


def test_slice_layers_exception_main_thread(layer_slicer):
    """Exception is raised on the main thread from an error on the main
    thread immediately when the task is created."""

    class FakeAsyncLayerError(FakeAsyncLayer):
        def _make_slice_request(self, dims) -> FakeSliceRequest:
            raise RuntimeError('_make_slice_request')

    layer = FakeAsyncLayerError()
    with pytest.raises(RuntimeError, match='_make_slice_request'):
        layer_slicer.slice_layers_async(layers=[layer], dims=Dims())


def test_slice_layers_exception_subthread_on_result(layer_slicer):
    """Exception is raised on the main thread from an error on a subthread
    only after result is called, not upon submission of the task."""

    class FakeAsyncLayerError(FakeAsyncLayer):
        def _get_slice(self, request: FakeSliceRequest) -> FakeSliceResponse:
            raise RuntimeError('_get_slice')

    layer = FakeAsyncLayerError()
    future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    done, _ = wait([future], timeout=5)
    if done:
        with pytest.raises(RuntimeError, match='_get_slice'):
            future.result()
    else:
        raise TimeoutError('Test future did not complete within timeout.')


def test_wait_until_idle(layer_slicer, single_threaded_executor):
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        slice_future = layer_slicer.slice_layers_async(
            layers=[layer], dims=dims
        )
        _wait_until_running(slice_future)
        # The slice task has started, but has not finished yet
        # because we are holding the layer's slicing lock.
        assert len(layer_slicer._layers_to_task) > 0
        # We can't call wait_until_idle on this thread because we're
        # holding the layer's slice lock, so submit it to be executed
        # on another thread and also wait for it to start.
        wait_future = single_threaded_executor.submit(
            layer_slicer.wait_until_idle
        )
        _wait_until_running(wait_future)

    wait_future.result()
    assert len(layer_slicer._layers_to_task) == 0


def _wait_until_running(future: Future):
    while not future.running():
        continue


def test_layer_slicer_force_sync_on_sync_layer(layer_slicer):
    layer = FakeSyncLayer()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert future.result() == {}
    assert not layer_slicer._force_sync


def test_layer_slicer_force_sync_on_async_layer(layer_slicer):
    layer = FakeAsyncLayer()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert future.result() == {}

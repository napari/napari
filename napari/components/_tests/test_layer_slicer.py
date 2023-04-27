import time
from concurrent.futures import Future, wait
from dataclasses import dataclass
from threading import RLock, current_thread, main_thread
from typing import Any

import numpy as np
import pytest

from napari._tests.utils import DEFAULT_TIMEOUT_SECS, LockableData
from napari.components import Dims
from napari.components._layer_slicer import _LayerSlicer
from napari.layers import Image, Points

# The following fakes are used to control execution of slicing across
# multiple threads, while also allowing us to mimic real classes
# (like layers) in the code base. This allows us to assert state and
# conditions that may only be temporarily true at different stages of
# an asynchronous task.


@dataclass(frozen=True)
class FakeSliceResponse:
    id: int


@dataclass(frozen=True)
class FakeSliceRequest:
    id: int
    lock: RLock

    def __call__(self) -> FakeSliceResponse:
        assert current_thread() != main_thread()
        with self.lock:
            return FakeSliceResponse(id=self.id)


class FakeAsyncLayer:
    def __init__(self) -> None:
        self._slice_request_count: int = 0
        self.slice_count: int = 0
        self.lock: RLock = RLock()

    def _make_slice_request(self, dims: Dims) -> FakeSliceRequest:
        assert current_thread() == main_thread()
        self._slice_request_count += 1
        return FakeSliceRequest(id=self._slice_request_count, lock=self.lock)

    def _update_slice_response(self, response: FakeSliceResponse):
        self.slice_count = response.id

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1


class FakeSyncLayer:
    def __init__(self) -> None:
        self.slice_count: int = 0

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1


@pytest.fixture()
def layer_slicer():
    layer_slicer = _LayerSlicer()
    layer_slicer._force_sync = False
    yield layer_slicer
    layer_slicer.shutdown()


def test_submit_with_one_async_layer_no_block(layer_slicer):
    layer = FakeAsyncLayer()

    future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert _wait_for_result(future)[layer].id == 1
    assert _wait_for_result(future)[layer].id == 1


def test_submit_with_multiple_async_layer_no_block(layer_slicer):
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()

    future = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    assert _wait_for_result(future)[layer1].id == 1
    assert _wait_for_result(future)[layer2].id == 1


def test_submit_emits_ready_event_when_done(layer_slicer):
    layer = FakeAsyncLayer()
    event_result = None

    def on_done(event):
        nonlocal event_result
        event_result = event.value

    layer_slicer.events.ready.connect(on_done)

    future = layer_slicer.submit(layers=[layer], dims=Dims())
    actual_result = _wait_for_result(future)

    assert actual_result is event_result


def test_submit_with_one_sync_layer(layer_slicer):
    layer = FakeSyncLayer()
    assert layer.slice_count == 0

    future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert future is None


def test_submit_with_multiple_sync_layer(layer_slicer):
    layer1 = FakeSyncLayer()
    layer2 = FakeSyncLayer()
    assert layer1.slice_count == 0
    assert layer2.slice_count == 0

    future = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    assert layer1.slice_count == 1
    assert layer2.slice_count == 1
    assert future is None


def test_submit_with_mixed_layers(layer_slicer):
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    assert layer1.slice_count == 0
    assert layer2.slice_count == 0

    future = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    assert layer2.slice_count == 1
    assert _wait_for_result(future)[layer1].id == 1
    assert layer2 not in _wait_for_result(future)


def test_submit_lock_blocking(layer_slicer):
    dims = Dims()
    layer = FakeAsyncLayer()

    assert layer.slice_count == 0
    with layer.lock:
        blocked = layer_slicer.submit(layers=[layer], dims=dims)
        assert not blocked.done()

    assert _wait_for_result(blocked)[layer].id == 1


def test_submit_multiple_calls_cancels_pending(layer_slicer):
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        blocked = layer_slicer.submit(layers=[layer], dims=dims)
        _wait_until_running(blocked)
        pending = layer_slicer.submit(layers=[layer], dims=dims)
        assert not pending.running()
        layer_slicer.submit(layers=[layer], dims=dims)
        assert not blocked.done()

    assert pending.cancelled()


def test_submit_mixed_allows_sync_to_run(layer_slicer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    with layer1.lock:
        blocked = layer_slicer.submit(layers=[layer1], dims=dims)
        layer_slicer.submit(layers=[layer2], dims=dims)
        assert layer2.slice_count == 1
        assert not blocked.done()

    assert _wait_for_result(blocked)[layer1].id == 1


def test_submit_mixed_allows_sync_to_run_one_slicer_call(layer_slicer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    with layer1.lock:
        blocked = layer_slicer.submit(layers=[layer1, layer2], dims=dims)

        assert layer2.slice_count == 1
        assert not blocked.done()

    assert _wait_for_result(blocked)[layer1].id == 1


def test_submit_with_multiple_async_layer_with_all_locked(
    layer_slicer,
):
    """ensure that if only all layers are locked, none continue"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()

    with layer1.lock, layer2.lock:
        blocked = layer_slicer.submit(layers=[layer1, layer2], dims=dims)
        assert not blocked.done()

    assert _wait_for_result(blocked)[layer1].id == 1
    assert _wait_for_result(blocked)[layer2].id == 1


def test_submit_task_to_layers_lock(layer_slicer):
    """ensure that if only one layer has a lock, the non-locked layer
    can continue"""
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        task = layer_slicer.submit(layers=[layer], dims=dims)
        assert task in layer_slicer._layers_to_task.values()

    assert _wait_for_result(task)[layer].id == 1
    assert task not in layer_slicer._layers_to_task


def test_submit_exception_main_thread(layer_slicer):
    """Exception is raised on the main thread from an error on the main
    thread immediately when the task is created."""

    class FakeAsyncLayerError(FakeAsyncLayer):
        def _make_slice_request(self, dims) -> FakeSliceRequest:
            raise RuntimeError('_make_slice_request')

    layer = FakeAsyncLayerError()
    with pytest.raises(RuntimeError, match='_make_slice_request'):
        layer_slicer.submit(layers=[layer], dims=Dims())


def test_submit_exception_subthread_on_result(layer_slicer):
    """Exception is raised on the main thread from an error on a subthread
    only after result is called, not upon submission of the task."""

    @dataclass(frozen=True)
    class FakeSliceRequestError(FakeSliceRequest):
        def __call__(self) -> FakeSliceResponse:
            assert current_thread() != main_thread()
            raise RuntimeError('FakeSliceRequestError')

    class FakeAsyncLayerError(FakeAsyncLayer):
        def _make_slice_request(self, dims: Dims) -> FakeSliceRequestError:
            self._slice_request_count += 1
            return FakeSliceRequestError(
                id=self._slice_request_count, lock=self.lock
            )

    layer = FakeAsyncLayerError()
    future = layer_slicer.submit(layers=[layer], dims=Dims())

    done, _ = wait([future], timeout=DEFAULT_TIMEOUT_SECS)
    assert done, 'Test future did not complete within timeout.'
    with pytest.raises(RuntimeError, match='FakeSliceRequestError'):
        _wait_for_result(future)


def test_wait_until_idle(layer_slicer, single_threaded_executor):
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        slice_future = layer_slicer.submit(layers=[layer], dims=dims)
        _wait_until_running(slice_future)
        # The slice task has started, but has not finished yet
        # because we are holding the layer's slicing lock.
        assert len(layer_slicer._layers_to_task) > 0
        # We can't call wait_until_idle on this thread because we're
        # holding the layer's slice lock, so submit it to be executed
        # on another thread and also wait for it to start.
        wait_future = single_threaded_executor.submit(
            layer_slicer.wait_until_idle,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
        _wait_until_running(wait_future)

    _wait_for_result(wait_future)
    assert len(layer_slicer._layers_to_task) == 0


def test_force_sync_on_sync_layer(layer_slicer):
    layer = FakeSyncLayer()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert future is None
    assert not layer_slicer._force_sync


def test_force_sync_on_async_layer(layer_slicer):
    layer = FakeAsyncLayer()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        future = layer_slicer.submit(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert future is None


def test_submit_with_one_3d_image(layer_slicer):
    np.random.seed(0)
    data = np.random.rand(8, 7, 6)
    lockable_data = LockableData(data)
    layer = Image(data=lockable_data, multiscale=False)
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 8, 1), (0, 7, 1), (0, 6, 1)),
        point=(2, 0, 0),
    )

    with lockable_data.lock:
        future = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()

    layer_result = _wait_for_result(future)[layer]
    np.testing.assert_equal(layer_result.data, data[2, :, :])


def test_submit_with_one_3d_points(layer_slicer):
    """ensure that async slicing of points does not block"""
    np.random.seed(0)
    num_points = 100
    data = np.rint(2.0 * np.random.rand(num_points, 3))
    layer = Points(data=data)

    # Note: We are directly accessing and locking the _data of layer. This
    #       forces a block to ensure that the async slicing call returns
    #       before slicing is complete.
    lockable_internal_data = LockableData(layer._data)
    layer._data = lockable_internal_data
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 3, 1), (0, 3, 1), (0, 3, 1)),
        point=(1, 0, 0),
    )

    with lockable_internal_data.lock:
        future = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()


def test_submit_after_shutdown_raises():
    layer_slicer = _LayerSlicer()
    layer_slicer._force_sync = False
    layer_slicer.shutdown()
    with pytest.raises(RuntimeError):
        layer_slicer.submit(layers=[FakeAsyncLayer()], dims=Dims())


def _wait_until_running(future: Future):
    """Waits until the given future is running using a default finite timeout."""
    sleep_secs = 0.01
    total_sleep_secs = 0
    while not future.running():
        time.sleep(sleep_secs)
        total_sleep_secs += sleep_secs
        if total_sleep_secs > DEFAULT_TIMEOUT_SECS:
            raise TimeoutError(
                f'Future did not start running after a timeout of {DEFAULT_TIMEOUT_SECS} seconds.'
            )


def _wait_for_result(future: Future) -> Any:
    """Waits until the given future is finished using a default finite timeout, and returns its result."""
    return future.result(timeout=DEFAULT_TIMEOUT_SECS)

from dataclasses import dataclass
from threading import RLock, current_thread, main_thread

from napari.components import Dims
from napari.components._layer_slicer import _LayerSlicer

# cases to consider
# - single + multiple layers that supports async (all of layers do support async)
# - single + multiple layers that don't support async (all of layers do not support async)
# - mix of layers that do and don't support async
#
# behaviors we want to test
# scheduling logic of the slicer (and not correctness of the slice response value)
#
# - for layers that support async, the slice task should not be run on the main thread (we don't want to block the calling)
# - for layers that do not support async, slicing should always be done once the method returns
# - slice requests should be run on the main thread
# - pending tasks are cancelled (at least when the new task will slice all layers for the pending task)
#
# run all tests with:
# pytest napari/components/_tests/test_layer_slicer.py -svv


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


def test_slice_layers_async_with_one_async_layer():
    layer_slicer = _LayerSlicer()
    layer = FakeAsyncLayer()

    future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    assert future.result()[layer].id == 1

    # TODO: make slicer a fixture and shutdown on teardown, using yield
    layer_slicer.shutdown()


def test_slice_layers_async_emits_ready_event_when_done():
    layer_slicer = _LayerSlicer()
    layer = FakeAsyncLayer()
    event_result = None

    def on_done(event):
        nonlocal event_result
        event_result = event.value

    layer_slicer.events.ready.connect(on_done)

    future = layer_slicer.slice_layers_async(layers=[layer], dims=Dims())
    actual_result = future.result()

    assert actual_result is event_result

    # TODO: make slicer a fixture and shutdown on teardown, using yield
    layer_slicer.shutdown()


def test_slice_layers_async_with_one_sync_layer():
    layer_slicer = _LayerSlicer()
    layer = FakeSyncLayer()
    assert layer.slice_count == 0

    layer_slicer.slice_layers_async(layers=[layer], dims=Dims())

    assert layer.slice_count == 1

    # TODO: make slicer a fixture and shutdown on teardown, using yield
    layer_slicer.shutdown()


def test_slice_layers_async_multiple_calls_cancels_pending():
    layer_slicer = _LayerSlicer()
    dims = Dims()
    layer = FakeAsyncLayer()
    with layer.lock:
        blocked = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        pending = layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        layer_slicer.slice_layers_async(layers=[layer], dims=dims)
        assert not blocked.done()

    assert pending.cancelled()

    # TODO: make slicer a fixture and shutdown on teardown, using yield
    layer_slicer.shutdown()

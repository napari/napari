import heapq
import itertools
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Event, RLock, current_thread, main_thread
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from napari._tests.utils import DEFAULT_TIMEOUT_SECS, LockableData
from napari.components import Dims
from napari.components._layer_slicer import (
    _chunk_count,
    _LayerSlicer,
    _ReadLimit,
    _reads_storage,
)
from napari.layers import Image, Labels, Points
from napari.layers.base._slice import _next_request_id
from napari.utils.notifications import notification_manager

if TYPE_CHECKING:
    import weakref

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
    started: Event

    def __call__(self) -> FakeSliceResponse:
        assert current_thread() != main_thread()
        self.started.set()
        with self.lock:
            return FakeSliceResponse(id=self.id)


@dataclass(frozen=True)
class ChunkedSliceRequest(FakeSliceRequest):
    """A request that reads ``chunks`` Dask chunks."""

    chunks: int = 1

    def chunk_count(self) -> int:
        return self.chunks


def _read_chunks(layer: 'FakeAsyncLayer', chunks: int):
    """Make the layer's requests read ``chunks`` chunks."""

    def make_slice_request(dims):
        layer._slice_request_count += 1
        return ChunkedSliceRequest(
            layer._slice_request_count, layer.lock, layer.started, chunks
        )

    return make_slice_request


@dataclass(frozen=True)
class StorageSliceRequest(FakeSliceRequest):
    """A request that reads from storage."""

    def reads_storage(self) -> bool:
        return True


def _read_storage(layer: 'FakeAsyncLayer'):
    """Make the layer's requests read from storage."""

    def make_slice_request(dims):
        layer._slice_request_count += 1
        return StorageSliceRequest(
            layer._slice_request_count, layer.lock, layer.started
        )

    return make_slice_request


class FakeAsyncLayer:
    def __init__(self) -> None:
        self._last_slice_id: int = 0
        self._slice_request_count: int = 0
        self.slice_count: int = 0
        self.loaded: bool = True
        self.visible: bool = True
        self.lock: RLock = RLock()
        self.started = Event()

    def _make_slice_request(self, dims: Dims) -> FakeSliceRequest:
        assert current_thread() == main_thread()
        self._slice_request_count += 1
        return FakeSliceRequest(
            id=self._slice_request_count, lock=self.lock, started=self.started
        )

    def _update_slice_response(self, response: FakeSliceResponse):
        self.slice_count = response.id

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1

    def _set_unloaded_slice_id(self, slice_id: int) -> None:
        self._last_slice_id = slice_id
        self.loaded = False

    def _update_loaded_slice_id(self, slice_id: int) -> None:
        if self._last_slice_id == slice_id:
            self.loaded = True

    @property
    def _slicing_state(self):
        return self


class FakeSyncLayer:
    def __init__(self) -> None:
        self.slice_count: int = 0
        self.visible: bool = True

    def _slice_dims(self, *args, **kwargs) -> None:
        self.slice_count += 1

    @property
    def _slicing_state(self):
        return self


class FixedReadLimit:
    """Lets a fixed number of layers read at once."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def start(self, layers: int) -> None:
        return None


class RecordingReadLimit(FixedReadLimit):
    """A fixed limit that records the reads it measures."""

    def __init__(self, limit: int) -> None:
        super().__init__(limit)
        self.started: list[int] = []
        self.finished = 0

    def start(self, layers: int) -> int:
        self.started.append(layers)
        return len(self.started)

    def finish(self, read: int) -> None:
        self.finished += 1


@pytest.fixture
def layer_slicer():
    from napari.settings import get_settings

    # Queue-order tests hold the sole worker while replacing waiting requests.
    get_settings().experimental.async_slicing_workers = 1
    layer_slicer = _LayerSlicer()
    layer_slicer._force_sync = False
    yield layer_slicer
    layer_slicer.shutdown()


def test_submit_with_one_async_layer_no_block(layer_slicer):
    layer = FakeAsyncLayer()

    [future] = layer_slicer.submit(layers=[layer], dims=Dims())

    assert _wait_for_response(future)[layer].id == 1


def test_submit_with_multiple_async_layer_no_block(layer_slicer):
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()

    future1, future2 = layer_slicer.submit(
        layers=[layer1, layer2], dims=Dims()
    )

    assert list(_wait_for_response(future1)) == [layer1]
    assert list(_wait_for_response(future2)) == [layer2]
    assert _wait_for_response(future1)[layer1].id == 1
    assert _wait_for_response(future2)[layer2].id == 1


def test_submit_emits_ready_event_when_done(layer_slicer):
    layer = FakeAsyncLayer()
    event_result = None

    def on_done(event):
        nonlocal event_result
        event_result = event.value

    layer_slicer.events.ready.connect(on_done)

    [future] = layer_slicer.submit(layers=[layer], dims=Dims())
    actual_result = _wait_for_result(future)

    assert actual_result is event_result


def test_submit_with_one_sync_layer(layer_slicer):
    layer = FakeSyncLayer()
    assert layer.slice_count == 0

    futures = layer_slicer.submit(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert futures == []


def test_submit_with_multiple_sync_layer(layer_slicer):
    layer1 = FakeSyncLayer()
    layer2 = FakeSyncLayer()
    assert layer1.slice_count == 0
    assert layer2.slice_count == 0

    futures = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    assert layer1.slice_count == 1
    assert layer2.slice_count == 1
    assert futures == []


def test_submit_with_mixed_layers(layer_slicer):
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    assert layer1.slice_count == 0
    assert layer2.slice_count == 0

    [future] = layer_slicer.submit(layers=[layer1, layer2], dims=Dims())

    assert layer2.slice_count == 1
    assert _wait_for_response(future)[layer1].id == 1
    assert layer2 not in _wait_for_response(future)


def test_submit_lock_blocking(layer_slicer):
    dims = Dims()
    layer = FakeAsyncLayer()

    assert layer.slice_count == 0
    with layer.lock:
        [blocked] = layer_slicer.submit(layers=[layer], dims=dims)
        assert not blocked.done()

    assert _wait_for_response(blocked)[layer].id == 1


def test_submit_replaces_pending_request_after_worker_claims_previous(
    layer_slicer,
):
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        [blocked] = layer_slicer.submit(layers=[layer], dims=dims)
        assert layer.started.wait(DEFAULT_TIMEOUT_SECS)
        [pending] = layer_slicer.submit(layers=[layer], dims=dims)
        for _ in range(5):
            [replacement] = layer_slicer.submit(layers=[layer], dims=dims)
            assert replacement is pending
        assert not blocked.done()
        assert not pending.running()

    assert _wait_for_response(blocked)[layer].id == 1
    assert _wait_for_response(pending)[layer].id == 7


def test_submit_replaces_each_pending_request_in_queue_order(layer_slicer):
    dims = Dims()
    blocker = FakeAsyncLayer()
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()
    executed = []
    layer_slicer.events.ready.connect(
        lambda event: executed.extend(event.value)
    )

    with blocker.lock:
        layer_slicer.submit(layers=[blocker], dims=dims)
        assert blocker.started.wait(DEFAULT_TIMEOUT_SECS)
        pending1, pending2 = layer_slicer.submit(
            layers=[layer1, layer2], dims=dims
        )
        assert layer_slicer.submit(layers=[layer2], dims=dims) == [pending2]
        assert layer_slicer.submit(layers=[layer2, layer1], dims=dims) == [
            pending2,
            pending1,
        ]

    layer_slicer.wait_until_idle(DEFAULT_TIMEOUT_SECS)
    assert [weak_layer() for weak_layer in executed] == [
        blocker,
        layer1,
        layer2,
    ]
    assert _wait_for_response(pending1)[layer1].id == 2
    assert _wait_for_response(pending2)[layer2].id == 3


def test_submit_after_pending_future_cancelled(layer_slicer):
    dims = Dims()
    blocker = FakeAsyncLayer()
    layer = FakeAsyncLayer()

    with blocker.lock:
        layer_slicer.submit(layers=[blocker], dims=dims)
        assert blocker.started.wait(DEFAULT_TIMEOUT_SECS)
        [pending] = layer_slicer.submit(layers=[layer], dims=dims)
        assert pending.cancel()
        assert layer.loaded
        [replacement] = layer_slicer.submit(layers=[layer], dims=dims)
        assert not layer.loaded

    assert _wait_for_response(replacement)[layer].id == 2


def test_submit_skips_deleted_pending_layer(layer_slicer):
    dims = Dims()
    blocker = FakeAsyncLayer()
    layer = FakeAsyncLayer()
    started = layer.started

    with blocker.lock:
        layer_slicer.submit(layers=[blocker], dims=dims)
        assert blocker.started.wait(DEFAULT_TIMEOUT_SECS)
        [pending] = layer_slicer.submit(layers=[layer], dims=dims)
        del layer

    assert not _wait_for_result(pending)
    assert not started.is_set()


def test_submit_reentrant_unloaded_observer(layer_slicer, monkeypatch):
    dims = Dims()
    layer = FakeAsyncLayer()
    submitted = []
    responses = []
    layer_slicer.events.ready.connect(
        lambda event: responses.extend(event.value.values())
    )
    set_unloaded = layer._set_unloaded_slice_id

    def on_unloaded(slice_id):
        was_loaded = layer.loaded
        set_unloaded(slice_id)
        if was_loaded:
            [nested] = layer_slicer.submit(layers=[layer], dims=dims)
            submitted.append(nested)
            # The outer call must respect the newer request even when its
            # task finishes before the observer returns.
            layer_slicer.wait_until_idle(DEFAULT_TIMEOUT_SECS)

    monkeypatch.setattr(layer, '_set_unloaded_slice_id', on_unloaded)
    assert layer_slicer.submit(layers=[layer], dims=dims) == []

    [nested] = submitted
    response = _wait_for_response(nested)[layer]
    assert response.id == layer._last_slice_id == 2
    layer_slicer.wait_until_idle(DEFAULT_TIMEOUT_SECS)
    assert responses == [response]


def test_submit_mixed_allows_sync_to_run(layer_slicer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    with layer1.lock:
        [blocked] = layer_slicer.submit(layers=[layer1], dims=dims)
        layer_slicer.submit(layers=[layer2], dims=dims)
        assert layer2.slice_count == 1
        assert not blocked.done()

    assert _wait_for_response(blocked)[layer1].id == 1


def test_submit_mixed_allows_sync_to_run_one_slicer_call(layer_slicer):
    """ensure that a blocked async slice doesn't block sync slicing"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeSyncLayer()
    with layer1.lock:
        [blocked] = layer_slicer.submit(layers=[layer1, layer2], dims=dims)

        assert layer2.slice_count == 1
        assert not blocked.done()

    assert _wait_for_response(blocked)[layer1].id == 1


def test_submit_with_multiple_async_layer_with_all_locked(
    layer_slicer,
):
    """ensure that if only all layers are locked, none continue"""
    dims = Dims()
    layer1 = FakeAsyncLayer()
    layer2 = FakeAsyncLayer()

    with layer1.lock, layer2.lock:
        blocked1, blocked2 = layer_slicer.submit(
            layers=[layer1, layer2], dims=dims
        )
        assert not blocked1.done()
        assert not blocked2.done()

    assert _wait_for_response(blocked1)[layer1].id == 1
    assert _wait_for_response(blocked2)[layer2].id == 1


def test_submit_exception_main_thread(layer_slicer):
    """Exception is raised on the main thread from an error on the main
    thread immediately when the task is created."""

    class FakeAsyncLayerError(FakeAsyncLayer):
        def _make_slice_request(self, dims) -> FakeSliceRequest:
            raise RuntimeError('_make_slice_request')

    layer = FakeAsyncLayerError()
    with pytest.raises(RuntimeError, match='_make_slice_request'):
        layer_slicer.submit(layers=[layer], dims=Dims())


def test_failed_replacement_allows_sibling_and_retry(
    layer_slicer, monkeypatch
):
    """Exception is raised on the main thread from an error on a subthread
    only after result is called, not upon submission of the task."""
    errors = []
    error_received = Event()

    def _record_error(exctype, value, traceback, thread=None):
        errors.append((exctype, value))
        error_received.set()

    monkeypatch.setattr(notification_manager, 'receive_error', _record_error)

    @dataclass(frozen=True)
    class FakeSliceRequestError(FakeSliceRequest):
        def __call__(self) -> FakeSliceResponse:
            assert current_thread() != main_thread()
            raise RuntimeError('FakeSliceRequestError')

    class FakeAsyncLayerError(FakeAsyncLayer):
        def _make_slice_request(self, dims: Dims) -> FakeSliceRequest:
            request = super()._make_slice_request(dims)
            if request.id != 2:
                return request
            return FakeSliceRequestError(
                id=request.id, lock=request.lock, started=request.started
            )

    layer = FakeAsyncLayerError()
    blocker = FakeAsyncLayer()
    sibling = FakeAsyncLayer()
    failures = []
    layer_slicer.events.failed.connect(
        lambda event: failures.append(event.request_id)
    )
    with blocker.lock:
        layer_slicer.submit(layers=[blocker], dims=Dims())
        assert blocker.started.wait(DEFAULT_TIMEOUT_SECS)
        future, success = layer_slicer.submit(
            layers=[layer, sibling], dims=Dims()
        )
        assert layer_slicer.submit(layers=[layer], dims=Dims()) == [future]

    assert _wait_for_response(success)[sibling].id == 1
    with pytest.raises(RuntimeError, match='FakeSliceRequestError'):
        _wait_for_response(future)
    assert error_received.wait(DEFAULT_TIMEOUT_SECS)
    assert failures == [2]
    assert layer.loaded
    assert len(errors) == 1
    assert errors[0][0] is RuntimeError
    assert str(errors[0][1]) == 'FakeSliceRequestError'

    [retry] = layer_slicer.submit(layers=[layer], dims=Dims())
    assert _wait_for_response(retry)[layer].id == 3


def test_wait_until_idle(layer_slicer, single_threaded_executor):
    dims = Dims()
    layer = FakeAsyncLayer()

    with layer.lock:
        [slice_future] = layer_slicer.submit(layers=[layer], dims=dims)
        assert layer.started.wait(DEFAULT_TIMEOUT_SECS)
        with pytest.raises(TimeoutError, match='Slicing 1 tasks'):
            layer_slicer.wait_until_idle(timeout=0)
        # We can't call wait_until_idle on this thread because we're
        # holding the layer's slice lock, so submit it to be executed
        # on another thread and also wait for it to start.
        wait_future = single_threaded_executor.submit(
            layer_slicer.wait_until_idle,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
        _wait_until_running(wait_future)

    _wait_for_result(wait_future)
    assert slice_future.done()


def test_layer_reads_one_request_at_a_time(layer_slicer, monkeypatch):
    """An idle worker does not start a second read of the same layer. The
    layer's waiting request is read after its current read, so a replaced
    request is skipped and responses keep request order."""
    dims = Dims()
    layer = FakeAsyncLayer()
    started = []

    def make_slice_request(dims):
        layer._slice_request_count += 1
        started.append(Event())
        return FakeSliceRequest(
            layer._slice_request_count, layer.lock, started[-1]
        )

    monkeypatch.setattr(layer, '_make_slice_request', make_slice_request)
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(layer_slicer, '_executor', executor)
        monkeypatch.setattr(layer_slicer, '_read_limit', FixedReadLimit(2))
        with layer.lock:
            [first] = layer_slicer.submit(layers=[layer], dims=dims)
            assert started[0].wait(DEFAULT_TIMEOUT_SECS)
            [second] = layer_slicer.submit(layers=[layer], dims=dims)
            [replaced] = layer_slicer.submit(layers=[layer], dims=dims)
            assert replaced is second
            assert not started[2].wait(0.2)
            with pytest.raises(TimeoutError, match='Slicing 2 tasks'):
                layer_slicer.wait_until_idle(timeout=0)

        layer_slicer.wait_until_idle(DEFAULT_TIMEOUT_SECS)

    assert _wait_for_response(first)[layer].id == 1
    assert _wait_for_response(second)[layer].id == 3
    assert not started[1].is_set()


def test_read_of_many_chunks_runs_alone(layer_slicer, monkeypatch):
    """A read of as many chunks as the shared Dask pool has threads keeps
    it busy, so the next layer waits for it instead of queueing in the pool
    behind it."""
    heavy, light = FakeAsyncLayer(), FakeAsyncLayer()
    monkeypatch.setattr(heavy, '_make_slice_request', _read_chunks(heavy, 8))
    monkeypatch.setattr(light, '_make_slice_request', _read_chunks(light, 1))
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(layer_slicer, '_executor', executor)
        monkeypatch.setattr(layer_slicer, '_read_limit', FixedReadLimit(2))
        monkeypatch.setattr(layer_slicer, '_chunk_budget', 8)
        with heavy.lock:
            _, light_read = layer_slicer.submit(
                layers=[heavy, light], dims=Dims()
            )
            assert heavy.started.wait(DEFAULT_TIMEOUT_SECS)
            assert not light.started.wait(0.2)
        assert _wait_for_response(light_read)[light].id == 1


def test_reads_of_few_chunks_share_the_pool(layer_slicer, monkeypatch):
    """Reads start while those in flight leave threads of the shared pool
    idle, up to the read limit."""
    layers = [FakeAsyncLayer() for _ in range(4)]
    for layer in layers:
        monkeypatch.setattr(
            layer, '_make_slice_request', _read_chunks(layer, 3)
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        monkeypatch.setattr(layer_slicer, '_executor', executor)
        monkeypatch.setattr(layer_slicer, '_read_limit', FixedReadLimit(4))
        monkeypatch.setattr(layer_slicer, '_chunk_budget', 7)
        with ExitStack() as stack:
            for layer in layers:
                stack.enter_context(layer.lock)
            futures = layer_slicer.submit(layers=layers, dims=Dims())
            for layer in layers[:3]:
                assert layer.started.wait(DEFAULT_TIMEOUT_SECS)
            assert not layers[3].started.wait(0.2)
        for layer, future in zip(layers, futures, strict=True):
            assert _wait_for_response(future)[layer].id == 1


def test_read_without_dask_chunks_passes_a_full_pool(
    layer_slicer, monkeypatch
):
    """A read of other data doesn't use the shared Dask pool, so it starts
    while a read of many chunks fills the pool, ahead of a Dask read that
    waits for it."""
    heavy, light, other = FakeAsyncLayer(), FakeAsyncLayer(), FakeAsyncLayer()
    monkeypatch.setattr(heavy, '_make_slice_request', _read_chunks(heavy, 8))
    monkeypatch.setattr(light, '_make_slice_request', _read_chunks(light, 1))
    with ThreadPoolExecutor(max_workers=3) as executor:
        monkeypatch.setattr(layer_slicer, '_executor', executor)
        monkeypatch.setattr(layer_slicer, '_read_limit', FixedReadLimit(3))
        monkeypatch.setattr(layer_slicer, '_chunk_budget', 8)
        with heavy.lock:
            _, light_read, other_read = layer_slicer.submit(
                layers=[heavy, light, other], dims=Dims()
            )
            assert heavy.started.wait(DEFAULT_TIMEOUT_SECS)
            assert _wait_for_response(other_read)[other].id == 1
            assert not light.started.is_set()
        assert _wait_for_response(light_read)[light].id == 1


@pytest.mark.parametrize(
    ('request_', 'expected'),
    [
        (object(), 0),
        (SimpleNamespace(chunk_count=lambda: 0), 0),
        (SimpleNamespace(chunk_count=lambda: 12), 12),
        (SimpleNamespace(chunk_count=lambda: 1 / 0), 1),
    ],
    ids=['no count', 'zero', 'twelve', 'error'],
)
def test_chunk_count_of_request(request_, expected):
    """Requests without a chunk count read no Dask chunks. One whose count
    fails counts as one, so it waits for the pool like a Dask read."""
    assert _chunk_count(request_) == expected


def test_read_limit_measures_only_reads_from_storage(
    layer_slicer, monkeypatch
):
    """A copy from memory says nothing about the storage, so the read limit
    neither measures it nor counts its layer among those with work."""
    stored, in_memory = FakeAsyncLayer(), FakeAsyncLayer()
    monkeypatch.setattr(stored, '_make_slice_request', _read_storage(stored))
    limit = RecordingReadLimit(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(layer_slicer, '_executor', executor)
        monkeypatch.setattr(layer_slicer, '_read_limit', limit)
        for future in layer_slicer.submit(
            layers=[stored, in_memory], dims=Dims()
        ):
            _wait_for_response(future)
    assert limit.started == [1]
    assert limit.finished == 1


@pytest.mark.parametrize(
    ('request_', 'expected'),
    [
        (object(), False),
        (SimpleNamespace(reads_storage=lambda: True), True),
        (SimpleNamespace(reads_storage=lambda: False), False),
        (SimpleNamespace(reads_storage=lambda: 1 / 0), True),
    ],
    ids=['unsaid', 'storage', 'memory', 'error'],
)
def test_reads_storage_of_request(request_, expected):
    """Requests that don't say read from memory, like those of points. One
    that fails to say counts as reading from storage."""
    assert _reads_storage(request_) is expected


def test_cancelled_request_leaves_no_second_reader(layer_slicer, monkeypatch):
    """A worker scheduled for a cancelled request does not start a second
    read of a layer whose next request is already being read."""
    dims = Dims()
    blockers = [FakeAsyncLayer(), FakeAsyncLayer()]
    layer = FakeAsyncLayer()
    started = []

    def make_slice_request(dims):
        layer._slice_request_count += 1
        started.append(Event())
        return FakeSliceRequest(
            layer._slice_request_count, layer.lock, started[-1]
        )

    monkeypatch.setattr(layer, '_make_slice_request', make_slice_request)
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(layer_slicer, '_executor', executor)
        monkeypatch.setattr(layer_slicer, '_read_limit', FixedReadLimit(2))
        layer.lock.acquire()
        for blocker in blockers:
            blocker.lock.acquire()
            layer_slicer.submit(layers=[blocker], dims=dims)
            assert blocker.started.wait(DEFAULT_TIMEOUT_SECS)
        try:
            [cancelled] = layer_slicer.submit(layers=[layer], dims=dims)
            assert cancelled.cancel()
            [second] = layer_slicer.submit(layers=[layer], dims=dims)
            blockers[0].lock.release()
            assert started[1].wait(DEFAULT_TIMEOUT_SECS)
            [third] = layer_slicer.submit(layers=[layer], dims=dims)
            blockers[1].lock.release()
            assert not started[2].wait(0.2)
        finally:
            for lock in (blockers[1].lock, layer.lock):
                if lock._is_owned():
                    lock.release()
        layer_slicer.wait_until_idle(DEFAULT_TIMEOUT_SECS)

    assert _wait_for_response(second)[layer].id == 2
    assert _wait_for_response(third)[layer].id == 3


def test_force_sync_on_sync_layer(layer_slicer):
    layer = FakeSyncLayer()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        futures = layer_slicer.submit(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert futures == []
    assert not layer_slicer._force_sync


def test_force_sync_on_async_layer(layer_slicer):
    layer = FakeAsyncLayer()

    with layer_slicer.force_sync():
        assert layer_slicer._force_sync
        futures = layer_slicer.submit(layers=[layer], dims=Dims())

    assert layer.slice_count == 1
    assert futures == []


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
        [future] = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()
    layer_result = _wait_for_response(future)[layer]
    np.testing.assert_equal(layer_result.image.view, data[2, :, :])


def test_submit_with_3d_labels(layer_slicer):
    np.random.seed(0)
    data = np.random.randint(20, size=(8, 7, 6))
    lockable_data = LockableData(data)
    layer = Labels(lockable_data, multiscale=False)
    dims = Dims(
        ndim=3,
        ndisplay=2,
        range=((0, 8, 1), (0, 7, 1), (0, 6, 1)),
        point=(2, 0, 0),
    )

    with lockable_data.lock:
        [future] = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()

    layer_result = _wait_for_response(future)[layer]
    np.testing.assert_equal(layer_result.image.view, data[2, :, :])


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
        [future] = layer_slicer.submit(layers=[layer], dims=dims)
        assert not future.done()


def test_shutdown_cancels_pending_replacements(layer_slicer, monkeypatch):
    dims = Dims()
    blocker = FakeAsyncLayer()
    pending_layer = FakeAsyncLayer()
    release = Event()

    class BlockingRequest(FakeSliceRequest):
        def __call__(self) -> FakeSliceResponse:
            self.started.set()
            assert release.wait(DEFAULT_TIMEOUT_SECS)
            return FakeSliceResponse(id=self.id)

    monkeypatch.setattr(
        blocker,
        '_make_slice_request',
        lambda dims: BlockingRequest(1, blocker.lock, blocker.started),
    )
    [active] = layer_slicer.submit(layers=[blocker], dims=dims)
    assert blocker.started.wait(DEFAULT_TIMEOUT_SECS)
    [pending] = layer_slicer.submit(layers=[pending_layer], dims=dims)
    assert layer_slicer.submit(layers=[pending_layer], dims=dims) == [pending]
    pending.add_done_callback(lambda _: release.set())

    layer_slicer.shutdown()

    layer_slicer.wait_until_idle(timeout=0)
    assert active.done()
    assert pending.cancelled()
    assert not pending_layer.started.is_set()


def test_submit_after_shutdown_raises():
    layer_slicer = _LayerSlicer()
    layer_slicer._force_sync = False
    layer_slicer.shutdown()
    with pytest.raises(RuntimeError):
        layer_slicer.submit(layers=[FakeAsyncLayer()], dims=Dims())


def test_slicers_present_slices_in_their_own_order():
    """Each view orders its slices, and an edit invalidates them in all views."""
    layer = Image(np.zeros((2, 2)))
    state = layer._slicing_state
    first, second = _LayerSlicer(), _LayerSlicer()
    try:
        older, newer = (
            SimpleNamespace(request_id=_next_request_id()) for _ in range(2)
        )
        state._set_unloaded_slice_id(newer.request_id)

        assert first._accept_slice_response(layer, newer)
        assert not first._accept_slice_response(layer, older)
        assert second._accept_slice_response(layer, older)
        assert first._can_publish_slice(layer, newer.request_id)
        assert second._can_publish_slice(layer, older.request_id)

        layer.data = np.ones((2, 2))

        assert not first._can_publish_slice(layer, newer.request_id)
        assert not second._can_publish_slice(layer, older.request_id)
        assert not second._accept_slice_response(layer, newer)
    finally:
        first.shutdown()
        second.shutdown()


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


# if remove quotes once we are Python 3.9+
def _wait_for_result(future: 'Future[Any]') -> Any:
    """Waits until the given future is finished returns its result."""
    return future.result(timeout=DEFAULT_TIMEOUT_SECS)


# remove quotes in types once we are python 3.9+ only.
def _wait_for_response(
    task: 'Future[dict[weakref.ReferenceType[Any], Any]]',
) -> dict:
    """Waits until the given slice task is finished and returns its result."""
    weak_result = _wait_for_result(task)
    result = {}
    for weak_layer, response in weak_result.items():
        if layer := weak_layer():
            result[layer] = response
    return result


class _GatedExecutor(ThreadPoolExecutor):
    """Holds submitted calls until the gate opens."""

    def __init__(self, max_workers: int) -> None:
        super().__init__(max_workers=max_workers)
        self.gate = Event()

    def submit(self, fn, /, *args, **kwargs):
        def run():
            self.gate.wait(DEFAULT_TIMEOUT_SECS)
            return fn(*args, **kwargs)

        return super().submit(run)


def _replace_before_claim(layer_slicer, monkeypatch, admitted, read):
    """Admit a read of ``admitted`` chunks and replace its request by one of
    ``read`` chunks before a worker claims it."""
    layer = FakeAsyncLayer()
    counts = iter([admitted, read])

    def make_slice_request(dims):
        layer._slice_request_count += 1
        return ChunkedSliceRequest(
            layer._slice_request_count, layer.lock, layer.started, next(counts)
        )

    monkeypatch.setattr(layer, '_make_slice_request', make_slice_request)
    executor = _GatedExecutor(max_workers=2)
    monkeypatch.setattr(layer_slicer, '_executor', executor)
    monkeypatch.setattr(layer_slicer, '_read_limit', FixedReadLimit(2))
    monkeypatch.setattr(layer_slicer, '_chunk_budget', 8)
    [first] = layer_slicer.submit(layers=[layer], dims=Dims())
    [second] = layer_slicer.submit(layers=[layer], dims=Dims())
    assert second is first
    return layer, first, executor.gate


def test_replaced_request_counts_the_chunks_it_reads(
    layer_slicer, monkeypatch
):
    """A request that replaces an admitted one before a worker claims it
    fills the shared pool, so the next Dask read waits for it."""
    light = FakeAsyncLayer()
    monkeypatch.setattr(light, '_make_slice_request', _read_chunks(light, 1))
    heavy, heavy_read, gate = _replace_before_claim(
        layer_slicer, monkeypatch, admitted=1, read=8
    )
    with heavy.lock:
        gate.set()
        assert heavy.started.wait(DEFAULT_TIMEOUT_SECS)
        [light_read] = layer_slicer.submit(layers=[light], dims=Dims())
        assert not light.started.wait(0.2)
    assert _wait_for_response(heavy_read)[heavy].id == 2
    assert _wait_for_response(light_read)[light].id == 1


def test_replaced_request_of_fewer_chunks_frees_the_pool(
    layer_slicer, monkeypatch
):
    """A request of few chunks that replaces an admitted one of many lets a
    waiting Dask read start once a worker claims it."""
    light = FakeAsyncLayer()
    monkeypatch.setattr(light, '_make_slice_request', _read_chunks(light, 1))
    heavy, heavy_read, gate = _replace_before_claim(
        layer_slicer, monkeypatch, admitted=8, read=1
    )
    with heavy.lock:
        [light_read] = layer_slicer.submit(layers=[light], dims=Dims())
        gate.set()
        assert heavy.started.wait(DEFAULT_TIMEOUT_SECS)
        assert light.started.wait(DEFAULT_TIMEOUT_SECS)
    assert _wait_for_response(heavy_read)[heavy].id == 2
    assert _wait_for_response(light_read)[light].id == 1


class SimulatedReads:
    """Reads for layers that always have a request waiting, on a fake clock."""

    def __init__(self, max_reads: int, layers: int = 4) -> None:
        self.now = 0.0
        self.read_limit = _ReadLimit(max_reads, clock=lambda: self.now)
        self.layers = layers
        self._in_flight: list = []
        self._order = itertools.count()

    def run(self, duration, reads: int) -> None:
        """Complete reads. One starting at a limit takes duration(limit) s."""
        for _ in range(reads):
            while len(self._in_flight) < min(
                self.read_limit.limit, self.layers
            ):
                read = self.read_limit.start(self.layers)
                finish = self.now + duration(read.limit)
                heapq.heappush(
                    self._in_flight, (finish, next(self._order), read)
                )
            self.now, _, read = heapq.heappop(self._in_flight)
            self.read_limit.finish(read)


def parallel(limit):
    return 0.1


def serial(limit):
    return 0.1 * limit


def test_read_limit_rises_on_parallel_storage():
    reads = SimulatedReads(4)
    assert reads.read_limit.limit == 2
    reads.run(parallel, 300)
    assert reads.read_limit._settled == 4


def test_read_limit_falls_on_serial_storage():
    reads = SimulatedReads(4)
    reads.run(serial, 300)
    assert reads.read_limit._settled == 1


def test_read_limit_compares_again_when_reads_slow_down():
    reads = SimulatedReads(4)
    reads.run(parallel, 300)
    assert reads.read_limit._settled == 4
    # Well before the periodic comparison.
    reads.run(serial, 150)
    assert reads.read_limit._settled == 1


def test_read_limit_rises_to_layers_with_work():
    """Between powers of two, the limit stops at the layers to read."""
    reads = SimulatedReads(16, layers=6)
    reads.run(parallel, 300)
    assert reads.read_limit._settled == 6


def test_read_limit_keeps_limit_without_enough_work():
    """With one layer to read, extra slots stay idle and teach nothing."""
    reads = SimulatedReads(4, layers=1)
    reads.run(parallel, 500)
    assert reads.read_limit._settled == 2


def test_read_limit_of_one_does_not_compare():
    reads = SimulatedReads(1)
    reads.run(serial, 500)
    assert reads.read_limit.limit == 1
    assert reads.read_limit._candidate is None


def test_read_limit_of_one_with_one_layer_does_not_compare():
    """Settled at one read, with one layer left to read, there is no other
    limit to compare with."""
    reads = SimulatedReads(4)
    reads.run(serial, 300)
    assert reads.read_limit._settled == 1
    reads.layers = 1
    reads.run(parallel, 500)
    assert reads.read_limit._settled == 1
    assert reads.read_limit._candidate is None

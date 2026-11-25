"""Handles the logic of asynchronously slicing of multiple layers.

See the NAP for more details: https://napari.org/dev/naps/4-async-slicing.html
"""

from __future__ import annotations

import logging
import math
import time
import weakref
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import fmean, variance
from threading import Condition
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from napari.layers import Layer
from napari.settings import get_settings
from napari.utils._dask_utils import _slice_pool_size
from napari.utils.events.event import EmitterGroup, Event

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from napari.components import Dims

logger = logging.getLogger('napari.components._layer_slicer')


# Layers that can be asynchronously sliced must be able to make
# a slice request that can be called and will produce a slice
# response. The request and response types are coupled but will
# vary per layer type, which means that the values of the dictionary
# result of ``_slice_layer`` cannot be fixed to a single type.


class _SliceRequest(Protocol):
    id: int

    def __call__(self) -> Any: ...


@runtime_checkable
class _AsyncSliceable(Protocol):
    """The methods needed for async slicing to be supported on a layer.

    These methods are private to avoid inflating the public API of
    layers while async slicing is being developed.
    """

    def _make_slice_request(self, dims: Dims) -> _SliceRequest:
        """Makes a callable slice request that returns a response.

        This method should run quickly, as it is expected to run on the main thread.
        Slower parts of slicing should be moved into the callable request, which can
        be run off the main thread.
        The request should capture any state it needs from a layer to generate
        the response and should not modify any of that state. In combination with
        other design choices, this allows us to avoid using locks while slicing.
        """

    def _update_slice_response(self, response: Any) -> None:
        """Passes through a completed slice response.

        This method should run on the main thread and is mostly needed to update
        slice state on layers.
        """

    def _set_unloaded_slice_id(self, slice_id: int) -> None:
        """Sets the ID associated with the latest slice request.

        This is needed to support ``Layer.loaded`` in async slicing.
        This could be done at the end of ``_make_slice_request``, but was separated
        to avoid mutations in that method and to clarify responsibilities.
        """


@dataclass
class _SliceTask:
    layer: weakref.ReferenceType[Layer]
    request: _SliceRequest
    future: Future[dict] = field(default_factory=Future)


@dataclass
class _Read:
    """A read in flight, as the read limit saw it start."""

    limit: int
    started: float
    turn: int
    full: bool  # every slot was busy when the read started
    layers: int  # layers reading from storage or waiting for a slot


class _ReadLimit:
    """Choose how many layers read at once, up to ``max_reads``.

    Storage that serves one read at a time (a single file handle, a library
    lock) gains nothing from parallel reads, and a read handed to a worker
    before the storage is free reads an outdated plane. The limit moves to
    where displayed slices stay freshest.

    A displayed slice is about as old as its read took, plus half the time
    until its layer's next read lands. With ``m`` layers taking turns and
    ``n`` reads of mean duration ``d`` in flight, reads complete at ``n / d``
    per second (Little's law), so that age is about ``d * (1 + m / (2 * n))``.
    Parallel storage keeps ``d`` as ``n`` grows. Serial storage stretches it
    in proportion, so an extra slot only adds waiting. Read durations show
    that queueing even where the number of reads per second hardly changes.
    Only reads from storage count: copies from memory come round more often
    with more slots and would pull ``d`` down.

    Starting from two reads, the limit is compared with double the number,
    up to the layers that have work, and if that does not help, with half,
    and moves whenever the other predicts an age shorter by ``MIN_GAIN``. The two limits take turns, so
    both see the same part of the workload, and only reads that started with
    every slot busy and finished in their turn count. Between comparisons,
    the limit is compared again after ``PERIOD`` reads, or sooner when reads
    at the chosen limit clearly slow down or speed up.

    .NET's thread pool chooses its number of threads similarly, by comparing
    neighboring counts (PortableThreadPool.HillClimbing.cs in dotnet/runtime).

    Callers hold the slicer's lock.
    """

    MIN_WINDOW = 12  # measured reads per limit before deciding
    # Measured reads per limit at which the estimate decides even if the
    # difference doesn't stand out from the spread of read durations.
    MAX_WINDOW = 48
    CONFIDENCE = 1.0  # standard errors between the estimate and the threshold
    MIN_GAIN = 0.05  # relative age improvement needed to change the limit
    SHIFT = 1.5  # change in read duration at the chosen limit that re-compares
    SHIFT_WINDOW = 32  # latest reads compared with all since the choice
    PERIOD = 384  # completed reads at the chosen limit between comparisons

    def __init__(
        self, max_reads: int, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.max_reads = max_reads
        self.limit = self._settled = min(2, max_reads)
        self._clock = clock
        self._candidate: int | None = None
        self._active = 0
        self._turn = 0
        self._started = 0
        self._reads = 0
        # Durations of measured reads per limit, and the layers they saw.
        self._durations: dict[int, list[float]] = {}
        self._layers = 0
        # Layers reading or waiting when the latest read started.
        self._busy = max_reads
        # Read durations at the chosen limit since it was chosen, and the
        # latest of them.
        self._total = 0.0
        self._count = 0
        self._recent: deque[float] = deque(maxlen=self.SHIFT_WINDOW)
        if max_reads > 1:
            self._compare(1 if self._settled < max_reads else -1)

    def start(self, layers: int) -> _Read:
        """Record a read starting while ``layers`` layers read or wait."""
        self._active += 1
        self._busy = layers
        read = _Read(
            self.limit,
            self._clock(),
            self._turn,
            self._active == self.limit,
            layers,
        )
        if self._candidate is not None:
            self._started += 1
            # A turn lasts long enough that most of its reads finish in it.
            if self._started >= 3 * max(self._settled, self._candidate):
                self.limit = (
                    self._candidate
                    if self.limit == self._settled
                    else self._settled
                )
                self._turn += 1
                self._started = 0
        return read

    def finish(self, read: _Read) -> None:
        """Record a read finishing, and adjust the limit if it is time."""
        self._active -= 1
        self._reads += 1
        if self._candidate is None:
            if self.max_reads == 1:
                return
            # Compare with more reads if more layers have work, else with
            # fewer, if there are fewer.
            down = -1 if self._settled > 1 else 0
            up = 1 if self._settled < self._ceiling() else down
            shift = self._shift(read)
            if shift or self._reads >= self.PERIOD:
                # Slower reads suggest the storage is busy: try fewer first.
                step = down if shift > 0 and down else up
                if step:
                    self._compare(step)
            return
        if read.full and read.turn == self._turn:
            self._durations.setdefault(read.limit, []).append(
                self._clock() - read.started
            )
            self._layers += read.layers
            wins = self._verdict()
            if wins is not None:
                self._decide(wins)
        elif self._reads >= 4 * self.MAX_WINDOW:
            # Too few reads with every slot busy: the slots outnumber the
            # work, so there is nothing to learn about the storage.
            self._decide(None)

    def _shift(self, read: _Read) -> int:
        """1 or -1 if reads at the chosen limit got clearly slower or faster.

        The latest reads are compared with all reads since the limit was
        chosen, so a change of storage or data is noticed before the next
        periodic comparison. Netflix's Gradient2Limit compares short- and
        long-term latency in the same way.
        """
        if not read.full or not self._count:
            return 0
        duration = self._clock() - read.started
        self._recent.append(duration)
        self._total += duration
        self._count += 1
        if len(self._recent) < self.SHIFT_WINDOW:
            return 0
        ratio = fmean(self._recent) * self._count / max(self._total, 1e-9)
        return 1 if ratio > self.SHIFT else -1 if ratio < 1 / self.SHIFT else 0

    def _verdict(self) -> bool | None:
        """Whether the candidate wins, or None while the reads can't tell.

        Read durations vary a lot from plane to plane. A comparison continues
        until the difference stands out from that spread, or until it has so
        many reads that the remaining doubt costs little either way.
        """
        n, c = self._settled, self._candidate
        a = self._durations.get(n, ())
        b = self._durations.get(c, ())
        if min(len(a), len(b)) < self.MIN_WINDOW:
            return None
        mean_a, mean_b = max(fmean(a), 1e-9), max(fmean(b), 1e-9)
        m = self._layers / (len(a) + len(b))
        # Log of the current limit's display age over the candidate's.
        gain = math.log(
            mean_a * (1 + m / (2 * n)) / (mean_b * (1 + m / (2 * c)))
        )
        margin = gain - math.log(1 + self.MIN_GAIN)
        # Standard error of the gain, from the relative spread of durations.
        spread = (
            variance(a) / mean_a**2 * (len(a) - 1)
            + variance(b) / mean_b**2 * (len(b) - 1)
        ) / (len(a) + len(b) - 2)
        noise = math.sqrt(spread * (1 / len(a) + 1 / len(b)))
        if (
            abs(margin) < self.CONFIDENCE * noise
            and min(len(a), len(b)) < self.MAX_WINDOW
        ):
            return None
        return margin > 0

    def _ceiling(self) -> int:
        """The most reads that can be busy, given the layers with work.

        Slots beyond those layers stay idle, so a comparison with them never
        sees every slot busy and learns nothing.
        """
        return max(1, min(self.max_reads, self._busy))

    def _neighbor(self, limit: int, step: int) -> int:
        """Double the limit, or halve it to a power of two.

        Like the multiplicative steps of TCP congestion control, this
        reaches large limits in a few comparisons, and neighbouring limits
        differ enough to tell apart quickly.
        """
        if step > 0:
            return min(2 * limit, self._ceiling())
        return 1 << ((limit - 1).bit_length() - 1)

    def _compare(self, step: int) -> None:
        self._candidate = self._neighbor(self._settled, step)
        self.limit = self._settled
        self._durations = {}
        self._layers = self._reads = self._started = 0
        self._turn += 1

    def _decide(self, wins: bool | None) -> None:
        n, c = self._settled, self._candidate
        logger.debug('Read limit %s against %s: %s', n, c, wins)
        self._candidate = None
        self._reads = 0
        self._turn += 1
        self.limit = n
        self._recent.clear()
        if wins is None or c is None:
            return
        durations = self._durations.get(c if wins else n, ())
        self._total, self._count = sum(durations), len(durations)
        if c > n:
            if wins:
                self._settled = self.limit = c
                if c < self._ceiling():
                    self._compare(1)
            elif n > 1:
                self._compare(-1)
        elif wins:
            self._settled = self.limit = c
            if c > 1:
                self._compare(-1)


def _chunk_count(request: _SliceRequest) -> int:
    """Dask chunks a request reads, 0 for other data, 1 if counting fails."""
    count = getattr(request, 'chunk_count', None)
    if count is None:
        return 0
    try:
        return max(0, count())
    except Exception:  # noqa: BLE001 - the read itself reports errors
        return 1


def _reads_storage(request: _SliceRequest) -> bool:
    """Whether a request reads from storage, not just from memory.

    Requests that don't say, such as those of points or vectors, read from
    memory. True if asking fails.
    """
    reads_storage = getattr(request, 'reads_storage', None)
    if reads_storage is None:
        return False
    try:
        return bool(reads_storage())
    except Exception:  # noqa: BLE001 - the read itself reports errors
        return True


class _LayerSlicer:
    """
    High level class to control the creation of a slice (via a slice request),
    submit it (synchronously or asynchronously) to a thread pool, and emit the
    results when complete.

    Events
    ------
    ready
        emitted after a layer is sliced with a dict value that maps from the
        layer to its slice response. Note that this may be emitted on the main or
        a non-main thread. If usage of this event relies on something happening
        on the main thread, actions should be taken to ensure that the callback
        is also executed on the main thread (e.g. by emitting a Qt signal from
        the callback, as `QtViewer._emit_slice_ready` does).
    failed
        emitted with a weak layer reference and request ID when a slice ends
        without a response. A GUI marks the event handled when it schedules
        completion on the main thread. Otherwise the slicer completes it here.
    """

    def __init__(self) -> None:
        self.events = EmitterGroup(source=self, ready=Event, failed=Event)
        settings = get_settings().experimental
        self._executor: Executor = ThreadPoolExecutor(
            max_workers=settings.async_slicing_workers
        )
        self._force_sync = not settings.async_
        self._shutdown = False
        # The condition protects queued requests and unfinished tasks. Once a
        # worker takes a task out of _pending, its request is immutable.
        # A layer has at most one read in flight, so its slices are read and
        # delivered in request order. Workers parallelize across layers, as
        # many at once as the read limit and the shared Dask pool allow.
        self._tasks_changed = Condition()
        self._pending: dict[weakref.ReferenceType[Layer], _SliceTask] = {}
        self._reading: set[weakref.ReferenceType[Layer]] = set()
        self._tasks: dict[Future[dict], _SliceTask] = {}
        # Layers with a waiting request and no read in flight, in turn order.
        self._ready: deque[weakref.ReferenceType[Layer]] = deque()
        self._read_limit = _ReadLimit(settings.async_slicing_workers)
        # Dask chunks of each read in flight. Another read starts only while
        # they leave threads of Dask's shared slicing pool idle.
        self._reading_chunks: dict[weakref.ReferenceType[Layer], int] = {}
        # Layers reading from storage, the reads the read limit measures.
        self._reading_storage: set[weakref.ReferenceType[Layer]] = set()
        self._chunk_budget = _slice_pool_size()
        # Latest request presented per layer, used only on the main thread.
        # The order of presented slices belongs to the slicer's view, while
        # edits invalidate slices on the layer itself, for every view.
        self._presented: weakref.WeakKeyDictionary[Layer, int] = (
            weakref.WeakKeyDictionary()
        )

    @contextmanager
    def force_sync(self) -> Generator[None, None, None]:
        """Context manager to temporarily force slicing to be synchronous.

        This should only be used from the main thread.

        >>> layer_slicer = _LayerSlicer()
        >>> layer = Image(...)  # an async-ready layer
        >>> with layer_slicer.force_sync():
        >>>     layer_slicer.submit(layers=[layer], dims=Dims())
        """
        prev = self._force_sync
        self._force_sync = True
        try:
            yield None
        finally:
            self._force_sync = prev

    def wait_until_idle(self, timeout: float | None = None) -> None:
        """Wait for all slicing tasks to complete before returning.

        Attributes
        ----------
        timeout: float or None
            (Optional) time in seconds to wait before raising TimeoutError. If set as None,
            there is no limit to the wait time. Defaults to None

        Raises
        ------
        TimeoutError: when the timeout limit has been exceeded and the task is
            not yet complete
        """
        with self._tasks_changed:
            if not self._tasks_changed.wait_for(
                lambda: not self._tasks, timeout=timeout
            ):
                raise TimeoutError(
                    f'Slicing {len(self._tasks)} tasks did not complete within timeout ({timeout}s).'
                )

    def submit(
        self,
        *,
        layers: Iterable[Layer],
        dims: Dims,
        force: bool = False,
    ) -> list[Future[dict]]:
        """Slices the given layers with the given dims.

        Each layer is sliced in its own task. A layer has at most one read in
        progress and one waiting request. A newer request replaces the waiting
        one in place, keeping its turn, and is read once the layer's current
        read finishes. A layer's responses therefore arrive in request order,
        but a response can be superseded by the time it arrives.

        This should only be called from the main thread.

        Parameters
        ----------
        layers : iterable of layers
            The layers to slice.
        dims : Dims
            The dimensions values associated with the view to be sliced.
        force : bool
            True if slicing should be forced to occur, even when some cache thinks
            it already has a valid slice ready. False otherwise.

        Returns
        -------
        list of futures
            One future per async layer scheduled by this call. Requests that
            replace a pending request share its future, whose result maps from
            a weak reference to the layer to the response of the request the
            worker claimed.
        """
        logger.debug(
            '_LayerSlicer.submit: layers=%s, dims=%s, force=%s',
            layers,
            dims,
            force,
        )

        # Not all layer types will initially be asynchronously sliceable.
        # The following logic gives us a way to handle those in the short
        # term as we develop, and also in the long term if there are cases
        # when we want to perform sync slicing anyway.
        tasks: dict[weakref.ReferenceType[Layer], Future[dict]] = {}
        sync_layers = []
        for layer in layers:
            # Slicing of non-visible layers is handled differently by sync
            # and async slicing. For async, we do not make request since a
            # later change to visibility triggers slicing. For sync, we want
            # to set the slice input with `Layer._slice_dims` but don't want
            # to fetch data yet (only if/when it becomes visible in the future).
            # Further development should allow us to remove this special case
            # by making the sync and async slicing code paths almost identical.
            slicing_state = layer._slicing_state
            if (
                isinstance(slicing_state, _AsyncSliceable)
                and not self._force_sync
                and layer.visible
            ):
                weak_layer = weakref.ref(layer)
                if weak_layer in tasks:
                    continue
                if self._shutdown:
                    raise RuntimeError(
                        'cannot schedule new slices after shutdown'
                    )
                logger.debug('Making async slice request for %s', layer)
                request = slicing_state._make_slice_request(dims)
                slicing_state._set_unloaded_slice_id(request.id)
                # A loaded-state observer can submit again before we queue
                # this request. Keep the newer request in that case.
                if slicing_state._last_slice_id != request.id:
                    continue
                try:
                    tasks[weak_layer] = self._queue_request(
                        weak_layer, request
                    )
                except BaseException:
                    self._finish_without_response(weak_layer, request.id)
                    raise
            else:
                logger.debug('Sync slicing for %s', layer)
                sync_layers.append(layer)

        # Synchronous slicing can run concurrently with the async tasks.
        for layer in sync_layers:
            layer._slice_dims(
                dims=dims,
                force=force,
            )

        return list(tasks.values())

    def _queue_request(
        self,
        weak_layer: weakref.ReferenceType[Layer],
        request: _SliceRequest,
    ) -> Future[dict]:
        with self._tasks_changed:
            if self._shutdown:
                raise RuntimeError('cannot schedule new slices after shutdown')
            task = self._pending.get(weak_layer)
            if task is not None:
                task.request = request
                logger.debug('Replaced pending request with %s', request.id)
                return task.future

            task = _SliceTask(weak_layer, request)
            self._pending[weak_layer] = task
            self._tasks[task.future] = task
            if weak_layer not in self._reading:
                self._ready.append(weak_layer)
                self._dispatch()
        task.future.add_done_callback(self._on_slice_done)
        return task.future

    def _accept_slice_response(self, layer: Layer, response: Any) -> bool:
        """Claim an async response before callbacks can deliver a newer one."""
        if response.request_id <= max(
            layer._slicing_state._discard_through_slice_id,
            self._presented.get(layer, -1),
        ):
            return False
        self._presented[layer] = response.request_id
        return True

    def _can_publish_slice(self, layer: Layer, request_id: int) -> bool:
        """Check again after notifying listeners that may edit or reslice."""
        return (
            self._presented.get(layer) == request_id
            and request_id > layer._slicing_state._discard_through_slice_id
        )

    def shutdown(self) -> None:
        """Shuts this down, preventing any new slice tasks from being submitted.

        This waits for any running tasks to finish, cancels any pending tasks,
        and disconnects any observers from this LayerSlicer's events.
        This should only be called from the main thread.
        """
        logger.debug('_LayerSlicer.shutdown')
        with self._tasks_changed:
            if self._shutdown:
                return
            self._shutdown = True
            queued = list(self._pending.values())
        for task in queued:
            task.future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.events.disconnect()
        self.events.ready.disconnect()
        self.events.failed.disconnect()

    def _dispatch(self) -> None:
        """Hand ready layers to workers while the read limit allows."""
        index = 0
        while (
            index < len(self._ready)
            and len(self._reading) < self._read_limit.limit
            and not self._shutdown
        ):
            weak_layer = self._ready[index]
            task = self._pending.get(weak_layer)
            # A cancelled request can leave its layer queued twice.
            if weak_layer in self._reading or task is None:
                del self._ready[index]
                continue
            # A read of many chunks already fills the pool, and another read
            # of Dask chunks would only queue behind it, older by the time it
            # lands. Reads of other data don't use the pool and pass it.
            chunks = _chunk_count(task.request)
            if (
                chunks
                and sum(self._reading_chunks.values()) >= self._chunk_budget
            ):
                index += 1
                continue
            del self._ready[index]
            self._reading.add(weak_layer)
            self._reading_chunks[weak_layer] = chunks
            self._executor.submit(self._slice_layer, weak_layer)

    def _slice_layer(self, weak_layer: weakref.ReferenceType[Layer]) -> None:
        """Claim the layer's latest pending request and slice it."""
        with self._tasks_changed:
            task = self._pending.pop(weak_layer, None)
            if task is None or not task.future.set_running_or_notify_cancel():
                self._release(weak_layer, None)
                return
            request = task.request
            # A copy from memory says nothing about the storage. Cheap reads
            # also come round more often with more slots, which would make
            # more slots look faster on storage that serves one read at once.
            read = None
            if _reads_storage(request):
                self._reading_storage.add(weak_layer)
                read = self._read_limit.start(self._storage_layers())
            # A newer request can replace the admitted one before a worker
            # claims it, so count the chunks of the request it reads.
            chunks = _chunk_count(request)
            if chunks != self._reading_chunks[weak_layer]:
                self._reading_chunks[weak_layer] = chunks
                self._dispatch()
        try:
            result = {}
            if weak_layer() is not None:
                result = {weak_layer: request()}
                self.events.ready(value=result)
        except BaseException as exc:  # noqa: BLE001 - delivered via the future
            self._finish_reading(weak_layer, read)
            task.future.set_exception(exc)
        else:
            self._finish_reading(weak_layer, read)
            task.future.set_result(result)

    def _storage_layers(self) -> int:
        """Layers reading from storage or waiting for a slot to."""
        waiting = sum(
            1
            for weak_layer in self._ready
            if (task := self._pending.get(weak_layer)) is not None
            and _reads_storage(task.request)
        )
        return len(self._reading_storage) + waiting

    def _finish_reading(
        self, weak_layer: weakref.ReferenceType[Layer], read: _Read | None
    ) -> None:
        """Hand the layer's next request, if any, to a worker."""
        with self._tasks_changed:
            self._release(weak_layer, read)

    def _release(
        self, weak_layer: weakref.ReferenceType[Layer], read: _Read | None
    ) -> None:
        self._reading.discard(weak_layer)
        self._reading_chunks.pop(weak_layer, None)
        self._reading_storage.discard(weak_layer)
        if read is not None:
            self._read_limit.finish(read)
        if weak_layer in self._pending:
            self._ready.append(weak_layer)
        self._dispatch()

    def _on_slice_done(self, future: Future[dict]) -> None:
        with self._tasks_changed:
            task = self._tasks.pop(future)
            if self._pending.get(task.layer) is task:
                del self._pending[task.layer]
            self._tasks_changed.notify_all()

        if future.cancelled():
            if not self._shutdown:
                self._finish_without_response(task.layer, task.request.id)
        elif (exception := future.exception()) is not None:
            self._finish_without_response(task.layer, task.request.id)

            from napari.utils.notifications import notification_manager

            notification_manager.receive_error(
                type(exception), exception, exception.__traceback__
            )

    def _finish_without_response(
        self, weak_layer: weakref.ReferenceType[Layer], request_id: int
    ) -> None:
        event = self.events.failed(layer=weak_layer, request_id=request_id)
        if not event.handled and (layer := weak_layer()) is not None:
            layer._slicing_state._update_loaded_slice_id(request_id)

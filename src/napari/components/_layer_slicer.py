"""Handles the logic of asynchronously slicing of multiple layers.

See the NAP for more details: https://napari.org/dev/naps/4-async-slicing.html
"""

from __future__ import annotations

import logging
import weakref
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from napari.layers import Layer
from napari.settings import get_settings
from napari.utils.events.event import EmitterGroup, Event

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

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
        self._executor: Executor = ThreadPoolExecutor(max_workers=1)
        self._force_sync = not get_settings().experimental.async_
        self._shutdown = False
        # The condition protects queued requests and unfinished tasks. Once a
        # worker takes a task out of _pending, its request is immutable.
        # A layer has at most one read in flight, so its slices are read and
        # delivered in request order. Workers parallelize across layers.
        self._tasks_changed = Condition()
        self._pending: dict[weakref.ReferenceType[Layer], _SliceTask] = {}
        self._reading: set[weakref.ReferenceType[Layer]] = set()
        self._tasks: dict[Future[dict], _SliceTask] = {}
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
                self._executor.submit(self._slice_layer, weak_layer)
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

    def _slice_layer(self, weak_layer: weakref.ReferenceType[Layer]) -> None:
        """Claim the layer's latest pending request and slice it."""
        with self._tasks_changed:
            # A worker scheduled for a cancelled request can find the layer
            # reading a newer one. That read hands on the next request.
            if weak_layer in self._reading:
                return
            task = self._pending.pop(weak_layer, None)
            if task is None or not task.future.set_running_or_notify_cancel():
                return
            self._reading.add(weak_layer)
            request = task.request
        try:
            result = {}
            if weak_layer() is not None:
                result = {weak_layer: request()}
                self.events.ready(value=result)
        except BaseException as exc:  # noqa: BLE001 - delivered via the future
            self._finish_reading(weak_layer)
            task.future.set_exception(exc)
        else:
            self._finish_reading(weak_layer)
            task.future.set_result(result)

    def _finish_reading(
        self, weak_layer: weakref.ReferenceType[Layer]
    ) -> None:
        """Hand the layer's next request, if any, to a worker."""
        with self._tasks_changed:
            self._reading.discard(weak_layer)
            if weak_layer in self._pending and not self._shutdown:
                self._executor.submit(self._slice_layer, weak_layer)

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

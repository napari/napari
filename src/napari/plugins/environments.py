"""Public lifecycle API for isolated plugin worker environments.

Plugin host code remains in napari's process and may use napari and Qt APIs.
Dependency-heavy functions are declared as qualified command targets and run
in persistent, per-plugin environments that napari prepares and owns.

Environment requirements belong in the npe2 manifest.
An embedded local worker package supplies importable worker code but must not
repeat runtime dependencies in its ``pyproject.toml``.
This contract keeps ordinary plugin installation lightweight and lets napari
present provisioning, cancellation, failure, worker, and cleanup state without
exposing the execution backend.

napari can enforce this split only for installation flows it manages.
Managed environments isolate dependencies from napari and other plugins, but
they are not security sandboxes: worker code retains the user's operating
system permissions.

napari retains a bounded, session-scoped history of operation events so a
management UI opened later can show provisioning output and structured
failures. Use :func:`list_plugin_environment_operations`,
:func:`add_plugin_environment_operation_callback`, and
:func:`clear_plugin_environment_operations` to read, observe, and clear it.
Backend completion messages describe sub-operations; only a terminal
:class:`PluginTaskState` means the complete napari-owned operation finished.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from uuid import uuid4
from weakref import WeakSet

from napari.utils.events import EmitterGroup, Event

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

T = TypeVar('T')
logger = logging.getLogger(__name__)


class PluginTaskState(Enum):
    """Lifecycle state of a managed plugin task."""

    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELED = 'canceled'

    @property
    def terminal(self) -> bool:
        """Return whether the task has reached a terminal state."""

        return self in {
            PluginTaskState.COMPLETED,
            PluginTaskState.FAILED,
            PluginTaskState.CANCELED,
        }


class PluginTaskPhase(Enum):
    """Current phase of a managed plugin task."""

    PREPARING = 'preparing'
    PROVISIONING = 'provisioning'
    STARTING = 'starting'
    EXECUTING = 'executing'
    CLEANING_UP = 'cleaning_up'


class PluginEnvironmentOperation(Enum):
    """Kind of managed plugin environment operation."""

    PREPARE = 'prepare'
    EXECUTE = 'execute'
    STOP = 'stop'
    REMOVE = 'remove'


class PluginEnvironmentState(Enum):
    """Persistent state of a declared managed plugin environment."""

    MISSING = 'missing'
    PREPARING = 'preparing'
    READY = 'ready'
    STALE = 'stale'
    FAILED = 'failed'


class PluginWorkerState(Enum):
    """Runtime worker state for a managed plugin environment."""

    STOPPED = 'stopped'
    RUNNING = 'running'
    STOPPING = 'stopping'


@dataclass(frozen=True)
class PluginTaskProgress:
    """Progress update emitted by a managed plugin task."""

    phase: PluginTaskPhase
    message: str
    current: int | None = None
    total: int | None = None


@dataclass(frozen=True)
class PluginEnvironmentInfo:
    """Backend-neutral snapshot of one managed plugin environment.

    ``state`` describes the persistent installation, while ``worker_state``
    describes lazily started runtime processes.
    """

    plugin: str
    environment_id: str
    display_name: str
    provision: str
    recipe_fingerprint: str | None
    state: PluginEnvironmentState
    worker_state: PluginWorkerState
    failure: str | None = None


@dataclass(frozen=True)
class PluginWorkerFailure:
    """Structured diagnostics for a failed worker command."""

    category: str | None
    message: str
    target: str | None = None
    traceback: str | None = None
    remote_exception_type: str | None = None
    remote_exception_message: str | None = None
    worker_environment: str | None = None
    worker_pid: int | None = None
    exit_code: int | None = None
    signal: int | None = None
    timeout: float | None = None
    elapsed: float | None = None
    serialization_context: str | None = None


@dataclass(frozen=True)
class PluginTaskMetadata:
    """Immutable identifiers used to present a managed plugin task."""

    operation: PluginEnvironmentOperation
    plugin: str | None = None
    environment_ids: tuple[str, ...] = ()
    command_id: str | None = None

    @property
    def environment_id(self) -> str | None:
        """Return the environment identifier when exactly one is selected."""

        return (
            self.environment_ids[0] if len(self.environment_ids) == 1 else None
        )


@dataclass(frozen=True)
class PluginEnvironmentOperationRecord:
    """Immutable event retained for a recent managed plugin operation."""

    sequence: int
    timestamp: datetime
    task_id: str
    operation: PluginEnvironmentOperation
    plugin: str | None
    environment_ids: tuple[str, ...]
    command_id: str | None
    phase: PluginTaskPhase | None
    state: PluginTaskState
    message: str
    current: int | None = None
    total: int | None = None
    details: str | None = None
    failure: PluginWorkerFailure | None = None

    @property
    def environment_id(self) -> str | None:
        """Return the environment identifier when exactly one is selected."""

        return (
            self.environment_ids[0] if len(self.environment_ids) == 1 else None
        )


@dataclass(frozen=True)
class _PluginTaskEvent:
    timestamp: datetime
    state: PluginTaskState
    phase: PluginTaskPhase | None
    message: str
    current: int | None = None
    total: int | None = None
    details: str | None = None
    failure: PluginWorkerFailure | None = None


class PluginEnvironmentError(RuntimeError):
    """Base exception for managed plugin environment failures."""

    def __init__(
        self,
        message: str,
        *,
        plugin: str | None = None,
        environment: str | None = None,
        command: str | None = None,
        phase: PluginTaskPhase | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.plugin = plugin
        self.environment = environment
        self.command = command
        self.phase = phase
        self.details = details
        if details:
            self.add_note(details)


class PluginEnvironmentUnavailableError(PluginEnvironmentError):
    """The managed environment backend is unavailable."""


class PluginEnvironmentProvisioningError(PluginEnvironmentError):
    """A plugin environment could not be prepared."""


class PluginWorkerError(PluginEnvironmentError):
    """A command failed in its managed worker."""

    def __init__(
        self,
        message: str,
        *,
        failure: PluginWorkerFailure | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.failure = failure


class PluginTaskCanceledError(PluginEnvironmentError):
    """A managed plugin task was canceled."""


def _immediate_dispatch(callback: Callable[[], None]) -> None:
    callback()


_task_dispatcher: Callable[[Callable[[], None]], None] = _immediate_dispatch
_task_observers: list[Callable[[PluginTask[Any]], Any]] = []
_task_hook_lock = threading.RLock()


def _set_task_dispatcher(
    dispatcher: Callable[[Callable[[], None]], None],
) -> None:
    """Install the application-specific event dispatcher."""

    global _task_dispatcher
    with _task_hook_lock:
        _task_dispatcher = dispatcher


def _add_task_observer(observer: Callable[[PluginTask[Any]], Any]) -> None:
    """Observe subsequently created tasks."""

    with _task_hook_lock:
        if observer not in _task_observers:
            _task_observers.append(observer)


def _remove_task_observer(
    observer: Callable[[PluginTask[Any]], Any],
) -> None:
    with _task_hook_lock:
        if observer in _task_observers:
            _task_observers.remove(observer)


def _notify_task_created(task: PluginTask[Any]) -> None:
    _operation_history.track(task)
    with _task_hook_lock:
        observers = tuple(_task_observers)
    for observer in observers:
        try:
            observer(task)
        except Exception:
            logger.exception('Plugin task observer failed')


class PluginTask(Generic[T]):
    """An observable and cancellable managed plugin operation.

    A task may be awaited or used from synchronous code through
    :meth:`result`. Use :meth:`add_progress_callback` and
    :meth:`add_done_callback` when callback delivery must be reliable even if
    a cached operation finishes immediately. :attr:`events` provides
    edge-triggered lifecycle events for existing napari event integrations.
    Treat :attr:`state`, rather than a backend progress message, as the
    authoritative completion signal.

    .. versionadded:: 0.8.1
    """

    def __init__(self, metadata: PluginTaskMetadata | None = None) -> None:
        self._metadata = metadata
        self._task_id = str(uuid4())
        self.events = EmitterGroup(
            source=self,
            started=Event,
            progress=Event,
            returned=Event,
            errored=Event,
            canceled=Event,
            finished=Event,
        )
        self.events.ignore_callback_errors = True
        self._state = PluginTaskState.PENDING
        self._phase: PluginTaskPhase | None = None
        self._progress: PluginTaskProgress | None = None
        self._result: T | None = None
        self._error: PluginEnvironmentError | None = None
        self._cancel_callback: Callable[[], Any] | None = None
        self._done_callbacks: list[Callable[[PluginTask[T]], Any]] = []
        self._progress_callbacks: list[
            Callable[[PluginTaskProgress], Any]
        ] = []
        self._progress_callback_sequences: dict[int, int] = {}
        self._progress_sequence = 0
        self._operation_events: deque[_PluginTaskEvent] = deque(maxlen=500)
        self._operation_callbacks: list[Callable[[_PluginTaskEvent], Any]] = []
        self._cancellation_requested = False
        self._future: Future[T] = Future()
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._record_operation_event(
            _PluginTaskEvent(
                timestamp=datetime.now(UTC),
                state=PluginTaskState.PENDING,
                phase=None,
                message='Managed plugin operation queued',
            )
        )

    @property
    def task_id(self) -> str:
        """Return the stable identifier for this task."""

        return self._task_id

    @property
    def metadata(self) -> PluginTaskMetadata | None:
        """Return immutable presentation metadata supplied by napari."""

        return self._metadata

    @property
    def state(self) -> PluginTaskState:
        """Return the current lifecycle state."""

        with self._lock:
            return self._state

    @property
    def phase(self) -> PluginTaskPhase | None:
        """Return the current operation phase."""

        with self._lock:
            return self._phase

    @property
    def progress(self) -> PluginTaskProgress | None:
        """Return the most recent progress update."""

        with self._lock:
            return self._progress

    @property
    def error(self) -> PluginEnvironmentError | None:
        """Return the terminal error, if any."""

        with self._lock:
            return self._error

    @property
    def cancellation_requested(self) -> bool:
        """Return whether cancellation has been requested."""

        with self._lock:
            return self._cancellation_requested

    @property
    def done(self) -> bool:
        """Return whether the task has finished."""

        return self._done.is_set()

    def add_done_callback(
        self,
        callback: Callable[[PluginTask[T]], Any],
    ) -> None:
        """Call ``callback`` once when the task finishes.

        If the task has already finished, the callback is dispatched
        immediately. This method is safe to use for fast cached operations.
        """

        with self._lock:
            if self._done.is_set():
                dispatch = True
            else:
                dispatch = False
                self._done_callbacks.append(callback)
        if dispatch:
            self._dispatch_callback(callback, self)

    def add_progress_callback(
        self,
        callback: Callable[[PluginTaskProgress], Any],
        *,
        replay: bool = True,
    ) -> None:
        """Observe progress, optionally replaying the most recent update."""

        with self._lock:
            if callback not in self._progress_callbacks:
                self._progress_callbacks.append(callback)
            progress = self._progress if replay else None
            sequence = self._progress_sequence
        if progress is not None:
            self._dispatch_progress_callback(callback, progress, sequence)

    def cancel(self) -> bool:
        """Request cancellation and return whether the request was accepted."""

        with self._lock:
            if self._state.terminal:
                return False
            first_request = not self._cancellation_requested
            self._cancellation_requested = True
            callback = self._cancel_callback
        if first_request and callback is not None:
            callback()
        return first_request

    def result(self, timeout: float | None = None) -> T:
        """Block until completion and return the result.

        Parameters
        ----------
        timeout : float, optional
            Maximum number of seconds to wait.
        """

        if not self._done.wait(timeout):
            raise TimeoutError(
                f'Plugin task did not finish within {timeout} seconds'
            )
        with self._lock:
            state = self._state
            result = self._result
            error = self._error
        if state is PluginTaskState.COMPLETED:
            return cast('T', result)
        if error is not None:
            raise error
        raise RuntimeError(
            f'Plugin task ended in unexpected state {state.value}'
        )

    def __await__(self) -> Generator[Any, None, T]:
        """Await the task without blocking the caller's event loop."""

        return self._async_result().__await__()

    async def _async_result(self) -> T:
        loop = asyncio.get_running_loop()
        waiter = asyncio.wrap_future(self._future, loop=loop)
        try:
            return await asyncio.shield(waiter)
        except asyncio.CancelledError:
            self.cancel()
            with suppress(PluginEnvironmentError, asyncio.CancelledError):
                await asyncio.shield(waiter)
            raise

    def _set_cancel_callback(self, callback: Callable[[], Any]) -> None:
        with self._lock:
            self._cancel_callback = callback
            requested = self._cancellation_requested
        if requested:
            callback()

    def _set_running(
        self,
        phase: PluginTaskPhase,
        message: str,
    ) -> None:
        with self._lock:
            if self._state is not PluginTaskState.PENDING:
                return
            self._state = PluginTaskState.RUNNING
        self._emit(self.events.started)
        self._report_progress(phase, message)

    def _report_progress(
        self,
        phase: PluginTaskPhase,
        message: str,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        progress = PluginTaskProgress(phase, message, current, total)
        with self._lock:
            if self._state.terminal:
                return
            self._phase = phase
            self._progress = progress
            self._progress_sequence += 1
            sequence = self._progress_sequence
            callbacks = tuple(self._progress_callbacks)
        self._record_operation_event(
            _PluginTaskEvent(
                timestamp=datetime.now(UTC),
                state=PluginTaskState.RUNNING,
                phase=phase,
                message=message,
                current=current,
                total=total,
            )
        )
        self._emit(self.events.progress, value=progress)
        for callback in callbacks:
            self._dispatch_progress_callback(callback, progress, sequence)

    def _set_result(self, result: T) -> None:
        with self._lock:
            if self._state.terminal:
                return
            if self._cancellation_requested:
                canceled = True
            else:
                canceled = False
                self._state = PluginTaskState.COMPLETED
                self._result = result
        if canceled:
            self._set_canceled()
            return
        self._record_operation_event(
            _PluginTaskEvent(
                timestamp=datetime.now(UTC),
                state=PluginTaskState.COMPLETED,
                phase=self.phase,
                message=self._completion_message(),
            )
        )
        self._future.set_result(result)
        try:
            self._emit(self.events.returned, value=result)
        finally:
            self._finish()

    def _set_error(self, error: PluginEnvironmentError) -> None:
        with self._lock:
            if self._state.terminal:
                return
            self._state = PluginTaskState.FAILED
            self._error = error
        self._record_operation_event(
            _PluginTaskEvent(
                timestamp=datetime.now(UTC),
                state=PluginTaskState.FAILED,
                phase=error.phase or self.phase,
                message=str(error),
                details=error.details,
                failure=(
                    error.failure
                    if isinstance(error, PluginWorkerError)
                    else None
                ),
            )
        )
        self._future.set_exception(error)
        try:
            self._emit(self.events.errored, value=error)
        finally:
            self._finish()

    def _set_canceled(self) -> None:
        error = PluginTaskCanceledError(
            'Plugin task was canceled',
            phase=self.phase,
        )
        with self._lock:
            if self._state.terminal:
                return
            self._state = PluginTaskState.CANCELED
            self._error = error
        self._record_operation_event(
            _PluginTaskEvent(
                timestamp=datetime.now(UTC),
                state=PluginTaskState.CANCELED,
                phase=self.phase,
                message='Managed plugin operation canceled',
            )
        )
        self._future.set_exception(error)
        try:
            self._emit(self.events.canceled)
        finally:
            self._finish()

    def _finish(self) -> None:
        with self._lock:
            self._done.set()
            callbacks = tuple(self._done_callbacks)
            self._done_callbacks.clear()
        for callback in callbacks:
            self._dispatch_callback(callback, self)
        self._emit(self.events.finished)

    @staticmethod
    def _emit(emitter: Callable[..., Any], **kwargs: Any) -> None:
        with _task_hook_lock:
            dispatcher = _task_dispatcher
        try:
            dispatcher(partial(emitter, **kwargs))
        except Exception:
            logger.exception('Plugin task event dispatch failed')

    @staticmethod
    def _dispatch_callback(callback: Callable[..., Any], value: Any) -> None:
        def invoke() -> None:
            try:
                callback(value)
            except Exception:
                logger.exception('Plugin task callback failed')

        with _task_hook_lock:
            dispatcher = _task_dispatcher
        try:
            dispatcher(invoke)
        except Exception:
            logger.exception('Plugin task callback dispatch failed')

    def _dispatch_progress_callback(
        self,
        callback: Callable[[PluginTaskProgress], Any],
        progress: PluginTaskProgress,
        sequence: int,
    ) -> None:
        def invoke() -> None:
            with self._lock:
                callback_id = id(callback)
                previous = self._progress_callback_sequences.get(
                    callback_id, -1
                )
                if sequence <= previous:
                    return
                self._progress_callback_sequences[callback_id] = sequence
            try:
                callback(progress)
            except Exception:
                logger.exception('Plugin task progress callback failed')

        with _task_hook_lock:
            dispatcher = _task_dispatcher
        try:
            dispatcher(invoke)
        except Exception:
            logger.exception('Plugin task callback dispatch failed')

    def _completion_message(self) -> str:
        if self._metadata is None:
            return 'Managed plugin operation completed'
        return {
            PluginEnvironmentOperation.PREPARE: 'Environment ready',
            PluginEnvironmentOperation.EXECUTE: 'Worker command completed',
            PluginEnvironmentOperation.STOP: 'Managed workers stopped',
            PluginEnvironmentOperation.REMOVE: 'Managed environments removed',
        }[self._metadata.operation]

    def _record_operation_event(self, event: _PluginTaskEvent) -> None:
        with self._lock:
            self._operation_events.append(event)
            callbacks = tuple(self._operation_callbacks)
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception('Plugin operation history callback failed')

    def _add_operation_callback(
        self,
        callback: Callable[[_PluginTaskEvent], Any],
        *,
        replay: bool = True,
    ) -> None:
        with self._lock:
            if callback not in self._operation_callbacks:
                self._operation_callbacks.append(callback)
            events = tuple(self._operation_events) if replay else ()
            for event in events:
                callback(event)


class _PluginOperationHistory:
    """Bounded, process-local history for managed environment operations."""

    def __init__(self, max_records: int = 2000) -> None:
        self._records: deque[PluginEnvironmentOperationRecord] = deque(
            maxlen=max_records
        )
        self._callbacks: list[
            Callable[[PluginEnvironmentOperationRecord], Any]
        ] = []
        self._tracked_tasks: WeakSet[PluginTask[Any]] = WeakSet()
        self._sequence = 0
        self._lock = threading.RLock()

    def track(self, task: PluginTask[Any]) -> None:
        """Retain existing and future events from a task exactly once."""

        metadata = task.metadata
        if metadata is None:
            return
        with self._lock:
            if task in self._tracked_tasks:
                return
            self._tracked_tasks.add(task)
        task_id = task.task_id

        def receive(event: _PluginTaskEvent) -> None:
            self._append(task_id, metadata, event)

        task._add_operation_callback(receive, replay=True)

    def list(
        self,
        plugin: str | None,
        environment_id: str | None,
    ) -> tuple[PluginEnvironmentOperationRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records
                if (plugin is None or record.plugin == plugin)
                and (
                    environment_id is None
                    or environment_id in record.environment_ids
                )
            )

    def add_callback(
        self,
        callback: Callable[[PluginEnvironmentOperationRecord], Any],
        *,
        replay: bool,
    ) -> Callable[[], None]:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
            records = tuple(self._records) if replay else ()
            for record in records:
                self._dispatch(callback, record)

        def unsubscribe() -> None:
            with self._lock, suppress(ValueError):
                self._callbacks.remove(callback)

        return unsubscribe

    def clear(
        self,
        plugin: str | None,
        environment_id: str | None,
    ) -> None:
        with self._lock:
            if plugin is None and environment_id is None:
                self._records.clear()
                return
            self._records = deque(
                (
                    record
                    for record in self._records
                    if not (
                        (plugin is None or record.plugin == plugin)
                        and (
                            environment_id is None
                            or environment_id in record.environment_ids
                        )
                    )
                ),
                maxlen=self._records.maxlen,
            )

    def _append(
        self,
        task_id: str,
        metadata: PluginTaskMetadata,
        event: _PluginTaskEvent,
    ) -> None:
        with self._lock:
            self._sequence += 1
            record = PluginEnvironmentOperationRecord(
                sequence=self._sequence,
                timestamp=event.timestamp,
                task_id=task_id,
                operation=metadata.operation,
                plugin=metadata.plugin,
                environment_ids=metadata.environment_ids,
                command_id=metadata.command_id,
                phase=event.phase,
                state=event.state,
                message=event.message,
                current=event.current,
                total=event.total,
                details=event.details,
                failure=event.failure,
            )
            self._records.append(record)
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            self._dispatch(callback, record)

    @staticmethod
    def _dispatch(
        callback: Callable[[PluginEnvironmentOperationRecord], Any],
        record: PluginEnvironmentOperationRecord,
    ) -> None:
        def invoke() -> None:
            try:
                callback(record)
            except Exception:
                logger.exception('Plugin operation observer failed')

        with _task_hook_lock:
            dispatcher = _task_dispatcher
        try:
            dispatcher(invoke)
        except Exception:
            logger.exception('Plugin operation observer dispatch failed')


_operation_history = _PluginOperationHistory()


def list_plugin_environment_operations(
    plugin: str | None = None,
    environment_id: str | None = None,
) -> tuple[PluginEnvironmentOperationRecord, ...]:
    """Return recent managed-operation events from this napari session.

    The history is bounded and includes task lifecycle state, progress,
    structured failures, and backend messages. It is retained independently
    of plugin-management windows so a window opened later can replay it.
    """

    return _operation_history.list(plugin, environment_id)


def add_plugin_environment_operation_callback(
    callback: Callable[[PluginEnvironmentOperationRecord], Any],
    *,
    replay: bool = False,
) -> Callable[[], None]:
    """Observe managed-operation records and return an unsubscribe callback."""

    return _operation_history.add_callback(callback, replay=replay)


def clear_plugin_environment_operations(
    plugin: str | None = None,
    environment_id: str | None = None,
) -> None:
    """Clear retained managed-operation records in the selected scope."""

    _operation_history.clear(plugin, environment_id)


def prepare_plugin_environment(environment_id: str) -> PluginTask[None]:
    """Prepare a declared plugin environment for later use.

    ``environment_id`` is the plugin-qualified identifier from the npe2
    manifest. Provisioned environments persist across napari sessions and are
    reused while their declared recipe remains unchanged.

    .. versionadded:: 0.8.1
    """

    from napari.plugins._environment_manager import (
        get_plugin_environment_manager,
    )

    task = get_plugin_environment_manager().prepare(environment_id)
    _notify_task_created(task)
    return task


def list_plugin_environments(
    plugin: str | None = None,
) -> tuple[PluginEnvironmentInfo, ...]:
    """Return declared and persistently owned plugin environments.

    Disabled plugins and environments retained after uninstallation remain
    visible so an installation UI can prepare or clean them without importing
    plugin host code.
    """

    from napari.plugins._environment_manager import (
        get_plugin_environment_manager,
    )

    return get_plugin_environment_manager().list_environments(plugin)


def stop_plugin_workers(
    plugin: str,
    environment_id: str | None = None,
) -> PluginTask[None]:
    """Stop active workers without deleting their persistent environments.

    Active work in the selected scope is canceled before worker resources are
    closed.
    """

    from napari.plugins._environment_manager import (
        get_plugin_environment_manager,
    )

    task = get_plugin_environment_manager().stop_workers(
        plugin, environment_id
    )
    _notify_task_created(task)
    return task


def remove_plugin_environments(
    plugin: str,
    environment_ids: Iterable[str] | None = None,
) -> PluginTask[None]:
    """Stop workers and remove persistently owned plugin environments.

    If ``environment_ids`` is omitted, all environments owned by ``plugin``
    are removed.
    Passing an empty iterable is a no-op.
    """

    from napari.plugins._environment_manager import (
        get_plugin_environment_manager,
    )

    task = get_plugin_environment_manager().remove_environments(
        plugin,
        None if environment_ids is None else tuple(environment_ids),
    )
    _notify_task_created(task)
    return task


def execute_worker_command(
    command_id: str,
    *args: Any,
    **kwargs: Any,
) -> PluginTask[Any]:
    """Execute a declared plugin worker command.

    Supported values are ``None``, booleans, numbers, strings, bytes, nested
    lists, tuples and dictionaries with scalar keys, and non-object NumPy
    arrays. The qualified command target executes outside the napari process.

    A command that declares ``accepts_worker_context`` receives the reserved
    ``napari_context`` keyword. The context exposes
    ``update(message, current=None, maximum=None)`` for progress and a
    ``cancel_requested`` boolean for cooperative cancellation. Worker modules
    should define this small protocol locally and must not import napari GUI
    APIs or Wetlands.

    Managed environments isolate dependencies; they are not security
    sandboxes. Worker code runs with the user's operating-system permissions.

    .. versionadded:: 0.8.1
    """

    from napari.plugins._environment_manager import (
        get_plugin_environment_manager,
    )

    task = get_plugin_environment_manager().execute(command_id, args, kwargs)
    _notify_task_created(task)
    return task


__all__ = (
    'PluginEnvironmentError',
    'PluginEnvironmentInfo',
    'PluginEnvironmentOperation',
    'PluginEnvironmentOperationRecord',
    'PluginEnvironmentProvisioningError',
    'PluginEnvironmentState',
    'PluginEnvironmentUnavailableError',
    'PluginTask',
    'PluginTaskCanceledError',
    'PluginTaskMetadata',
    'PluginTaskPhase',
    'PluginTaskProgress',
    'PluginTaskState',
    'PluginWorkerError',
    'PluginWorkerFailure',
    'PluginWorkerState',
    'add_plugin_environment_operation_callback',
    'clear_plugin_environment_operations',
    'execute_worker_command',
    'list_plugin_environment_operations',
    'list_plugin_environments',
    'prepare_plugin_environment',
    'remove_plugin_environments',
    'stop_plugin_workers',
)

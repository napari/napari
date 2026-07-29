"""Public API for executing plugin commands in managed environments."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from napari.utils.events import EmitterGroup, Event

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

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


@dataclass(frozen=True)
class PluginTaskProgress:
    """Progress update emitted by a managed plugin task."""

    phase: PluginTaskPhase
    message: str
    current: int | None = None
    total: int | None = None


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

    .. versionadded:: 0.8.1
    """

    def __init__(self) -> None:
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
        self._cancellation_requested = False
        self._future: Future[T] = Future()
        self._lock = threading.RLock()
        self._done = threading.Event()

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
    'PluginEnvironmentProvisioningError',
    'PluginEnvironmentUnavailableError',
    'PluginTask',
    'PluginTaskCanceledError',
    'PluginTaskPhase',
    'PluginTaskProgress',
    'PluginTaskState',
    'PluginWorkerError',
    'PluginWorkerFailure',
    'execute_worker_command',
    'prepare_plugin_environment',
)

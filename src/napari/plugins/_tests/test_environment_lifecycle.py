from __future__ import annotations

import gc
import threading
import weakref
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from napari.plugins import (
    _environment_manager as manager_module,
    environments as environment_api,
)
from napari.plugins._environment_manager import PluginEnvironmentManager
from napari.plugins._environment_types import BackendCanceled
from napari.plugins._tests.test_environments import (
    _command,
    _FakeBackend,
    _recipe,
)
from napari.plugins.environments import (
    PluginEnvironmentOperation,
    PluginEnvironmentUnavailableError,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskMetadata,
    PluginTaskPhase,
    PluginTaskState,
    PluginWorkerError,
    PluginWorkerFailure,
    _immediate_dispatch,
    _notify_task_created,
    _PluginOperationHistory,
    _set_task_dispatcher,
)

if TYPE_CHECKING:
    from pathlib import Path

    from napari.plugins._environment_types import (
        BackendPool,
        CancelCallbackSetter,
        ProgressCallback,
    )


class _BlockingStartBackend(_FakeBackend):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()

    def start_pool(
        self,
        environment: object,
        *,
        progress: ProgressCallback,
    ) -> BackendPool:
        self.start_entered.set()
        if not self.allow_start.wait(2):
            raise TimeoutError('test did not release pool startup')
        return super().start_pool(environment, progress=progress)


class _BlockingRemovalBackend(_FakeBackend):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.removal_started = threading.Event()
        self.removal_canceled = threading.Event()
        self.block_removal = False

    def remove_environment(
        self,
        physical_name: str,
        *,
        progress: ProgressCallback | None = None,
        set_cancel_callback: CancelCallbackSetter | None = None,
    ) -> None:
        if not self.block_removal:
            return super().remove_environment(
                physical_name,
                progress=progress,
                set_cancel_callback=set_cancel_callback,
            )
        self.removal_started.set()
        if set_cancel_callback is not None:
            set_cancel_callback(self.removal_canceled.set)
        if not self.removal_canceled.wait(2):
            raise TimeoutError('test did not cancel stale cleanup')
        raise BackendCanceled


def test_cancellation_requested_before_backend_callback_is_forwarded() -> None:
    task: PluginTask[None] = PluginTask()
    canceled = False

    def cancel_backend() -> None:
        nonlocal canceled
        canceled = True

    assert task.cancel()
    task._set_cancel_callback(cancel_backend)

    assert canceled
    assert not task.cancel()


def test_completed_task_replays_latest_progress_and_done_once() -> None:
    task: PluginTask[int] = PluginTask()
    task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
    task._report_progress(
        PluginTaskPhase.PROVISIONING,
        'Provisioned',
        1,
        1,
    )
    task._set_result(42)
    progress = []
    completed = []

    task.add_progress_callback(progress.append)
    task.add_done_callback(completed.append)

    assert [update.message for update in progress] == ['Provisioned']
    assert completed == [task]


def test_progress_replay_cannot_arrive_after_a_newer_update() -> None:
    queued_callbacks = []
    task: PluginTask[None] = PluginTask()
    received = []
    _set_task_dispatcher(queued_callbacks.append)
    try:
        task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
        task.add_progress_callback(received.append)
        task._report_progress(PluginTaskPhase.PROVISIONING, 'Provisioned')

        queued_callbacks[4]()
        queued_callbacks[2]()
    finally:
        _set_task_dispatcher(_immediate_dispatch)

    assert [update.message for update in received] == ['Provisioned']


def test_dispatcher_failure_does_not_change_task_outcome() -> None:
    task: PluginTask[int] = PluginTask()

    def fail_dispatch(callback) -> None:
        raise RuntimeError('dispatcher unavailable')

    _set_task_dispatcher(fail_dispatch)
    try:
        task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
        task._set_result(42)
    finally:
        _set_task_dispatcher(_immediate_dispatch)

    assert task.done
    assert task.result() == 42


def test_shutdown_cancels_active_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(tmp_path)
    backend.block_preparation = True
    recipe = _recipe()
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_environment',
        lambda environment_id: recipe,
    )

    task = manager.prepare(recipe.environment_id)
    assert backend.preparation_started.wait(2)

    manager.close()

    assert backend.preparation_canceled.is_set()
    assert backend.closed
    assert task.state is PluginTaskState.CANCELED
    with pytest.raises(PluginTaskCanceledError):
        task.result()


def test_closed_manager_rejects_new_tasks(tmp_path: Path) -> None:
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: _FakeBackend(root),
    )
    manager.close()

    task = manager.prepare('example-plugin.worker')

    assert task.state is PluginTaskState.FAILED
    with pytest.raises(
        PluginEnvironmentUnavailableError,
        match='manager is closed',
    ):
        task.result()


def test_shutdown_closes_pool_created_while_shutdown_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _BlockingStartBackend(tmp_path)
    command = _command('example-plugin.compute', _recipe())
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )
    task = manager.execute(command.command_id, (), {})
    assert backend.start_entered.wait(2)
    cancellation_requested = threading.Event()
    cancel = task.cancel

    def record_cancel() -> bool:
        result = cancel()
        cancellation_requested.set()
        return result

    task.cancel = record_cancel  # type: ignore[method-assign]
    shutdown = threading.Thread(target=manager.close)

    shutdown.start()
    assert cancellation_requested.wait(2)
    backend.allow_start.set()
    shutdown.join(2)

    assert not shutdown.is_alive()
    assert len(backend.started) == 1
    assert backend.started[0].closed
    assert backend.closed
    with pytest.raises(PluginTaskCanceledError):
        task.result()


def test_release_cancels_queued_plugin_task_before_pool_insertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(tmp_path)
    backend.block_preparation = True
    command = _command('plugin-a.compute', _recipe(plugin='plugin-a'))
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
        max_parallel_tasks=1,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )

    running = manager.execute(command.command_id, (), {})
    assert backend.preparation_started.wait(2)
    queued = manager.execute(command.command_id, (), {})

    manager.release_plugin(command.plugin)

    for task in (running, queued):
        with pytest.raises(PluginTaskCanceledError):
            task.result(2)
    assert backend.started == []
    assert manager._pool_entries == {}
    manager.close()


def test_stop_workers_preempts_blocked_executor_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(tmp_path)
    backend.block_preparation = True
    command = _command('plugin-a.compute', _recipe(plugin='plugin-a'))
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
        max_parallel_tasks=1,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )

    running = manager.execute(command.command_id, (), {})
    assert backend.preparation_started.wait(2)
    stopped = manager.stop_workers(
        command.plugin,
        command.environment_id,
    )

    with pytest.raises(PluginTaskCanceledError):
        running.result(2)
    assert stopped.result(2) is None
    assert backend.preparation_canceled.is_set()
    assert manager._pool_entries == {}
    manager.close()


def test_concurrent_control_tasks_do_not_cancel_each_other(
    tmp_path: Path,
) -> None:
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: _FakeBackend(root),
        max_parallel_tasks=1,
    )
    lifecycle_lock = manager._plugin_lifecycle_lock('plugin-a')
    lifecycle_lock.acquire()
    try:
        first = manager.stop_workers('plugin-a')
        second = manager.remove_environments('plugin-a')
    finally:
        lifecycle_lock.release()

    assert first.result(2) is None
    assert second.result(2) is None
    assert first.state is PluginTaskState.COMPLETED
    assert second.state is PluginTaskState.COMPLETED
    manager.close()


def test_execute_submitted_during_stop_is_canceled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(tmp_path)
    command = _command('plugin-a.compute', _recipe(plugin='plugin-a'))
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
        max_parallel_tasks=1,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )
    lifecycle_lock = manager._plugin_lifecycle_lock(command.plugin)
    lifecycle_lock.acquire()
    try:
        stopping = manager.stop_workers(command.plugin)
        execution = manager.execute(command.command_id, (), {})
    finally:
        lifecycle_lock.release()

    assert stopping.result(2) is None
    with pytest.raises(PluginTaskCanceledError):
        execution.result(2)
    assert backend.prepared == []
    assert backend.started == []
    manager.close()


def test_control_task_starts_in_cleanup_phase(tmp_path: Path) -> None:
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: _FakeBackend(root),
    )
    progress = []

    task = manager.stop_workers('plugin-a')
    task.add_progress_callback(progress.append)

    assert task.result(2) is None
    assert progress
    assert progress[0].phase is PluginTaskPhase.CLEANING_UP
    manager.close()


def test_stale_cleanup_reports_progress_and_can_be_canceled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _BlockingRemovalBackend(tmp_path)
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    selected = _recipe()
    monkeypatch.setattr(
        manager_module,
        '_find_environment',
        lambda environment_id: selected,
    )
    try:
        manager.prepare(selected.environment_id).result(2)
        selected = _recipe(requirement='example-dependency==2')
        backend.block_removal = True
        progress = []

        task = manager.prepare(selected.environment_id)
        task.add_progress_callback(progress.append)
        assert backend.removal_started.wait(2)

        assert not task.done
        assert progress[-1].phase is PluginTaskPhase.CLEANING_UP
        assert 'Removing previous environment' in progress[-1].message
        assert task.cancel()
        with pytest.raises(PluginTaskCanceledError):
            task.result(2)
        assert backend.removal_canceled.is_set()

        backend.block_removal = False
        assert manager.prepare(selected.environment_id).result(2) is None
        assert len(backend.known) == 1
    finally:
        manager.close()


def test_manager_tasks_have_immutable_presentation_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(tmp_path)
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    recipe = _recipe()
    command = _command('example-plugin.compute', recipe)
    monkeypatch.setattr(
        manager_module,
        '_owner_for_contribution',
        lambda contribution_id: recipe.plugin,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_environment',
        lambda environment_id: recipe,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )
    try:
        prepared = manager.prepare(recipe.environment_id)
        assert prepared.result(2) is None
        executed = manager.execute(command.command_id, (), {})
        executed.result(2)
        stopped = manager.stop_workers(recipe.plugin, recipe.environment_id)
        assert stopped.result(2) is None
        removed = manager.remove_environments(
            recipe.plugin, (recipe.environment_id,)
        )
        assert removed.result(2) is None

        assert prepared.metadata == PluginTaskMetadata(
            PluginEnvironmentOperation.PREPARE,
            recipe.plugin,
            (recipe.environment_id,),
        )
        assert executed.metadata == PluginTaskMetadata(
            PluginEnvironmentOperation.EXECUTE,
            recipe.plugin,
            (recipe.environment_id,),
            command.command_id,
        )
        assert stopped.metadata is not None
        assert stopped.metadata.operation is PluginEnvironmentOperation.STOP
        assert removed.metadata is not None
        assert removed.metadata.operation is PluginEnvironmentOperation.REMOVE
        with pytest.raises(FrozenInstanceError):
            prepared.metadata.plugin = 'changed'  # type: ignore[misc,union-attr]
    finally:
        manager.close()


def test_recent_operation_history_is_bounded_replayable_and_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _PluginOperationHistory(max_records=3)
    monkeypatch.setattr(environment_api, '_operation_history', history)
    metadata = PluginTaskMetadata(
        PluginEnvironmentOperation.EXECUTE,
        'example-plugin',
        ('example-plugin.worker',),
        'example-plugin.compute',
    )
    task: PluginTask[None] = PluginTask(metadata)
    received = []
    unsubscribe = environment_api.add_plugin_environment_operation_callback(
        received.append
    )

    _notify_task_created(task)
    task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
    task._report_progress(PluginTaskPhase.EXECUTING, 'Running', 1, 2)
    failure = PluginWorkerFailure(
        category='remote_exception',
        message='bad input',
        traceback='ValueError: bad input',
    )
    task._set_error(
        PluginWorkerError(
            'Worker failed',
            details='remote traceback',
            failure=failure,
        )
    )
    with pytest.raises(PluginWorkerError):
        task.result()

    records = environment_api.list_plugin_environment_operations(
        'example-plugin', 'example-plugin.worker'
    )
    assert len(records) == 3
    assert [record.sequence for record in records] == sorted(
        record.sequence for record in records
    )
    assert records[-1].state is PluginTaskState.FAILED
    assert records[-1].details == 'remote traceback'
    assert records[-1].failure == failure
    assert records[-1].timestamp.tzinfo is not None
    assert received[-1] == records[-1]

    replayed = []
    stop_replay = environment_api.add_plugin_environment_operation_callback(
        replayed.append, replay=True
    )
    assert replayed == list(records)
    stop_replay()
    unsubscribe()

    environment_api.clear_plugin_environment_operations(
        'example-plugin', 'example-plugin.worker'
    )
    assert environment_api.list_plugin_environment_operations() == ()


def test_operation_history_does_not_retain_tasks() -> None:
    history = _PluginOperationHistory(max_records=3)
    task: PluginTask[None] = PluginTask(
        PluginTaskMetadata(PluginEnvironmentOperation.PREPARE)
    )
    history.track(task)
    task_reference = weakref.ref(task)

    del task
    gc.collect()

    assert task_reference() is None
    assert len(history._tracked_tasks) == 0

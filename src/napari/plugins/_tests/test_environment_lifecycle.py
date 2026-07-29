from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from napari.plugins import _environment_manager as manager_module
from napari.plugins._environment_manager import PluginEnvironmentManager
from napari.plugins._tests.test_environments import (
    _command,
    _FakeBackend,
    _recipe,
)
from napari.plugins.environments import (
    PluginEnvironmentUnavailableError,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskPhase,
    PluginTaskState,
    _immediate_dispatch,
    _set_task_dispatcher,
)

if TYPE_CHECKING:
    from pathlib import Path

    from napari.plugins._environment_types import (
        BackendPool,
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

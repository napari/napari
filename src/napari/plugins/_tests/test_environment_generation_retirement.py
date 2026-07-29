from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import pytest

from napari.plugins import _environment_manager as manager_module
from napari.plugins._environment_manager import PluginEnvironmentManager
from napari.plugins._tests.test_environments import (
    _command,
    _FakeBackend,
    _FakePool,
    _recipe,
)
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskPhase,
)

if TYPE_CHECKING:
    from pathlib import Path

    from napari.plugins._environment_types import (
        CancelCallbackSetter,
        ProgressCallback,
    )


class _BlockingPool(_FakePool):
    def __init__(
        self,
        environment: str,
        *,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(environment)
        self._entered = entered
        self._release = release

    def execute(
        self,
        target: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        accepts_context: bool,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any:
        self._entered.set()
        if not self._release.wait(2):
            raise TimeoutError('test did not release the old generation')
        return super().execute(
            target,
            args,
            kwargs,
            accepts_context=accepts_context,
            progress=progress,
            set_cancel_callback=set_cancel_callback,
        )


class _FirstExecutionBlockingBackend(_FakeBackend):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.first_execution_entered = threading.Event()
        self.release_first_execution = threading.Event()

    def start_pool(
        self,
        environment: Any,
        *,
        progress: ProgressCallback,
    ) -> _FakePool:
        if self.started:
            return super().start_pool(environment, progress=progress)
        pool = _BlockingPool(
            environment,
            entered=self.first_execution_entered,
            release=self.release_first_execution,
        )
        self.started.append(pool)
        return pool


def _raise_callback_error(event: Any) -> None:
    raise RuntimeError(f'callback failed for {event.type}')


def test_raising_event_callbacks_do_not_prevent_task_completion() -> None:
    returned: PluginTask[int] = PluginTask()
    returned.events.progress.connect(_raise_callback_error)
    returned.events.returned.connect(_raise_callback_error)
    returned._set_running(PluginTaskPhase.PREPARING, 'Preparing')
    returned._set_result(42)

    assert returned.done
    assert returned.result() == 42

    failed: PluginTask[None] = PluginTask()
    failed.events.errored.connect(_raise_callback_error)
    expected_error = PluginEnvironmentError('worker failed')
    failed._set_running(PluginTaskPhase.EXECUTING, 'Executing')
    failed._set_error(expected_error)

    assert failed.done
    with pytest.raises(PluginEnvironmentError) as error:
        failed.result()
    assert error.value is expected_error

    canceled: PluginTask[None] = PluginTask()
    canceled.events.canceled.connect(_raise_callback_error)
    canceled._set_running(PluginTaskPhase.PROVISIONING, 'Provisioning')
    canceled._set_canceled()

    assert canceled.done
    with pytest.raises(PluginTaskCanceledError):
        canceled.result()


def test_sequential_recipe_change_closes_and_removes_old_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend(tmp_path)
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    selected = _command('example-plugin.compute', _recipe())
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: selected,
    )

    try:
        first_result = manager.execute(selected.command_id, (), {}).result(2)
        first_name = first_result['environment']
        first_pool = backend.started[0]
        selected = _command(
            selected.command_id,
            _recipe(requirement='example-dependency==2'),
        )

        second_result = manager.execute(selected.command_id, (), {}).result(2)

        assert second_result['environment'] != first_name
        assert first_pool.closed
        assert backend.removed == [first_name]
        assert first_name not in backend.known
    finally:
        manager.close()


def test_in_flight_generation_is_retired_after_its_lease_is_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FirstExecutionBlockingBackend(tmp_path)
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    selected = _command('example-plugin.compute', _recipe())
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: selected,
    )

    try:
        first_task = manager.execute(selected.command_id, (), {})
        assert backend.first_execution_entered.wait(2)
        first_name = backend.started[0].environment
        first_pool = backend.started[0]
        selected = _command(
            selected.command_id,
            _recipe(requirement='example-dependency==2'),
        )

        second_result = manager.execute(selected.command_id, (), {}).result(2)

        assert second_result['environment'] != first_name
        assert not first_pool.closed
        assert first_name in backend.known
        assert first_name not in backend.removed

        backend.release_first_execution.set()
        assert first_task.result(2)['environment'] == first_name
        assert first_pool.closed
        assert backend.removed == [first_name]
        assert first_name not in backend.known
    finally:
        backend.release_first_execution.set()
        manager.close()


def test_long_logical_ids_with_common_slug_prefix_do_not_share_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend(tmp_path)
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    common = 'example-plugin.' + ('long-segment-' * 8)
    recipe_a = _recipe(environment=f'{common}a')
    recipe_b = _recipe(environment=f'{common}b')
    command_a = _command('example-plugin.compute-a', recipe_a)
    command_b = _command('example-plugin.compute-b', recipe_b)
    selected = command_a
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: selected,
    )

    try:
        first_a = manager.execute(command_a.command_id, (), {}).result(2)
        selected = command_b
        result_b = manager.execute(command_b.command_id, (), {}).result(2)
        pool_b = backend.started[1]
        assert manager._physical_prefix(recipe_a) != manager._physical_prefix(
            recipe_b
        )

        selected = _command(
            command_a.command_id,
            _recipe(
                environment=recipe_a.environment_id,
                requirement='example-dependency==2',
            ),
        )
        second_a = manager.execute(command_a.command_id, (), {}).result(2)

        assert first_a['environment'] in backend.removed
        assert first_a['environment'] not in backend.known
        assert result_b['environment'] in backend.known
        assert result_b['environment'] not in backend.removed
        assert second_a['environment'] in backend.known
        assert not pool_b.closed
    finally:
        manager.close()


def test_manifest_iteration_excludes_disabled_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def iter_manifests(*, disabled: bool):
        calls.append(disabled)
        return iter(())

    monkeypatch.setattr(
        'npe2.plugin_manager.iter_manifests',
        iter_manifests,
    )

    assert manager_module._iter_manifests() == ()
    assert calls == [False]

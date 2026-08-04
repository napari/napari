from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from npe2 import PluginManifest

from napari.plugins import _environment_manager as manager_module, _npe2
from napari.plugins._environment_manager import PluginEnvironmentManager
from napari.plugins._environment_types import (
    BackendCanceled,
    BackendFailure,
    BackendProgress,
    EnvironmentRecipe,
    WorkerCommand,
)
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginEnvironmentProvisioningError,
    PluginEnvironmentState,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskPhase,
    PluginTaskState,
    PluginWorkerError,
    PluginWorkerState,
)

if TYPE_CHECKING:
    from pathlib import Path

    from napari.plugins._environment_types import (
        CancelCallbackSetter,
        ProgressCallback,
    )


def _recipe(
    plugin: str = 'example-plugin',
    environment: str = 'example-plugin.worker',
    requirement: str = 'example-dependency==1',
) -> EnvironmentRecipe:
    return EnvironmentRecipe(
        plugin=plugin,
        plugin_version='1.0',
        environment_id=environment,
        python='3.12',
        conda=(),
        pypi=(requirement,),
        channels=('conda-forge',),
        local_packages=(),
        lockfile=None,
    )


def _command(
    command_id: str,
    recipe: EnvironmentRecipe,
    target: str = 'worker:echo',
) -> WorkerCommand:
    return WorkerCommand(
        plugin=recipe.plugin,
        environment_id=recipe.environment_id,
        command_id=command_id,
        target=target,
        accepts_context=False,
        recipe=recipe,
    )


def _manifest_for_recipe(
    recipe: EnvironmentRecipe,
    *,
    display_name: str = 'Example worker',
    provision: str = 'on_demand',
) -> SimpleNamespace:
    environment = SimpleNamespace(
        id=recipe.environment_id,
        display_name=display_name,
        provision=provision,
        python=recipe.python,
        conda=list(recipe.conda),
        pypi=list(recipe.pypi),
        channels=list(recipe.channels),
        local_packages=[],
        lockfile=None,
    )
    return SimpleNamespace(
        name=recipe.plugin,
        package_version=recipe.plugin_version,
        contributions=SimpleNamespace(environments=[environment]),
    )


class _FakePool:
    def __init__(self, environment: str) -> None:
        self.environment = environment
        self.closed = False
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

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
        self.calls.append((target, args, kwargs))
        set_cancel_callback(lambda: None)
        progress(
            BackendProgress(
                PluginTaskPhase.EXECUTING,
                f'Running {target}',
                1,
                1,
            )
        )
        if target == 'worker:fail':
            raise BackendFailure(
                'remote worker failed',
                details='remote traceback: ValueError: bad input',
                diagnostics={
                    'category': 'remote_exception',
                    'message': 'bad input',
                    'target': target,
                    'traceback': 'ValueError: bad input',
                    'remote_exception_type': 'builtins.ValueError',
                    'remote_exception_message': 'bad input',
                    'worker_environment': self.environment,
                    'worker_pid': 123,
                    'exit_code': None,
                    'signal': None,
                    'timeout': None,
                    'elapsed': 0.1,
                    'serialization_context': None,
                },
            )
        return {
            'environment': self.environment,
            'args': args,
            'kwargs': kwargs,
            'accepts_context': accepts_context,
        }

    def close(self) -> None:
        self.closed = True


class _FakeBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prepared: list[str] = []
        self.started: list[_FakePool] = []
        self.removed: list[str] = []
        self.known: set[str] = set()
        self.fail_preparation = False
        self.fail_removal = False
        self.removal_failures = 0
        self.block_preparation = False
        self.preparation_started = threading.Event()
        self.preparation_canceled = threading.Event()
        self.closed = False

    def fingerprint(self, recipe: EnvironmentRecipe) -> str:
        payload = {
            'plugin': recipe.plugin,
            'environment': recipe.environment_id,
            'python': recipe.python,
            'conda': recipe.conda,
            'pypi': recipe.pypi,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def prepare_environment(
        self,
        physical_name: str,
        recipe: EnvironmentRecipe,
        *,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> str:
        self.prepared.append(physical_name)
        self.preparation_started.set()
        set_cancel_callback(self.preparation_canceled.set)
        progress(
            BackendProgress(
                PluginTaskPhase.PROVISIONING,
                f'Provisioning {recipe.environment_id}',
                1,
                2,
            )
        )
        if self.block_preparation:
            self.preparation_canceled.wait(5)
            raise BackendCanceled
        if self.fail_preparation:
            raise BackendFailure(
                'provisioning failed',
                details='pixi exited with status 1',
            )
        self.known.add(physical_name)
        return physical_name

    def start_pool(
        self,
        environment: Any,
        *,
        progress: ProgressCallback,
    ) -> _FakePool:
        progress(
            BackendProgress(
                PluginTaskPhase.STARTING,
                f'Starting {environment}',
            )
        )
        pool = _FakePool(environment)
        self.started.append(pool)
        return pool

    def remove_environment(
        self,
        physical_name: str,
        *,
        progress: ProgressCallback | None = None,
        set_cancel_callback: CancelCallbackSetter | None = None,
    ) -> None:
        if set_cancel_callback is not None:
            set_cancel_callback(lambda: None)
        if progress is not None:
            progress(
                BackendProgress(
                    PluginTaskPhase.CLEANING_UP,
                    f'Removing {physical_name}',
                )
            )
        if self.fail_removal or self.removal_failures:
            if self.removal_failures:
                self.removal_failures -= 1
            raise BackendFailure('removal failed', details='cleanup failed')
        self.removed.append(physical_name)
        self.known.discard(physical_name)

    def environment_names(self) -> tuple[str, ...]:
        return tuple(self.known)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def environment_manager(tmp_path: Path):
    backend = _FakeBackend(tmp_path)
    manager = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: backend,
    )
    yield manager, backend
    manager.close()


def test_task_reports_progress_and_result() -> None:
    task: PluginTask[int] = PluginTask()
    events: list[tuple[str, Any]] = []
    task.events.started.connect(lambda event: events.append(('started', None)))
    task.events.progress.connect(
        lambda event: events.append(('progress', event.value))
    )
    task.events.returned.connect(
        lambda event: events.append(('returned', event.value))
    )
    task.events.finished.connect(
        lambda event: events.append(('finished', None))
    )

    task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
    task._report_progress(PluginTaskPhase.PROVISIONING, 'Installing', 2, 3)
    task._set_result(42)

    assert task.result() == 42
    assert task.state is PluginTaskState.COMPLETED
    assert [event[0] for event in events] == [
        'started',
        'progress',
        'progress',
        'returned',
        'finished',
    ]
    assert events[2][1].current == 2


def test_backend_progress_uses_logical_environment_name() -> None:
    task: PluginTask[None] = PluginTask()
    received = []
    task._set_running(PluginTaskPhase.CLEANING_UP, 'Cleaning')
    task.add_progress_callback(received.append)
    physical_name = 'example-plugin-example-plugin.worker-a1b2-c3d4'

    PluginEnvironmentManager._report(
        task,
        BackendProgress(
            PluginTaskPhase.CLEANING_UP,
            f"Removing managed environment '{physical_name}'",
            1,
            2,
        ),
        physical_name=physical_name,
        environment_id='example-plugin.worker',
    )

    assert received[-1].message == (
        "Removing managed environment 'example-plugin.worker'"
    )
    assert received[-1].current == 1
    assert received[-1].total == 2
    task._set_result(None)
    assert task.result() is None


def test_incompatible_plugin_recipes_are_isolated(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    first = _command(
        'plugin-a.compute',
        _recipe(
            plugin='plugin-a',
            environment='plugin-a.worker',
            requirement='shared-dependency==1',
        ),
    )
    second = _command(
        'plugin-b.compute',
        _recipe(
            plugin='plugin-b',
            environment='plugin-b.worker',
            requirement='shared-dependency==2',
        ),
    )
    commands = {first.command_id: first, second.command_id: second}
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        commands.__getitem__,
    )

    first_result = manager.execute(first.command_id, (), {}).result()
    second_result = manager.execute(second.command_id, (), {}).result()

    assert first_result['environment'] != second_result['environment']
    assert len(backend.started) == 2
    assert not {'shared-dependency==1', 'shared-dependency==2'} & set(
        __import__('sys').modules
    )


def test_recipe_is_reused_and_change_selects_new_generation(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    selected = _command('example-plugin.compute', _recipe())
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: selected,
    )

    first = manager.execute(selected.command_id, (), {}).result()
    second = manager.execute(selected.command_id, (), {}).result()
    selected = _command(
        selected.command_id,
        _recipe(requirement='example-dependency==2'),
    )
    changed = manager.execute(selected.command_id, (), {}).result()

    assert first['environment'] == second['environment']
    assert len(backend.started) == 2
    assert changed['environment'] != first['environment']


def test_nested_values_and_arrays_cross_runtime_boundary(
    environment_manager, monkeypatch
) -> None:
    manager, _ = environment_manager
    command = _command('example-plugin.echo', _recipe())
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )
    array = np.arange(12).reshape(3, 4)
    value = {'image': array, 'metadata': ['plain', {'scale': 2.5}]}

    result = manager.execute(
        command.command_id, (value,), {'flag': True}
    ).result()

    returned = result['args'][0]
    np.testing.assert_array_equal(returned['image'], array)
    assert returned['metadata'] == value['metadata']
    assert result['kwargs'] == {'flag': True}


def test_preparation_progress_can_be_canceled(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    backend.block_preparation = True
    recipe = _recipe()
    monkeypatch.setattr(
        manager_module,
        '_find_environment',
        lambda environment_id: recipe,
    )

    task = manager.prepare(recipe.environment_id)
    assert backend.preparation_started.wait(2)
    assert task.cancel()

    with pytest.raises(PluginTaskCanceledError):
        task.result(2)
    assert backend.preparation_canceled.is_set()
    assert task.state is PluginTaskState.CANCELED


def test_provisioning_failure_has_plugin_context(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    backend.fail_preparation = True
    recipe = _recipe()
    monkeypatch.setattr(
        manager_module,
        '_find_environment',
        lambda environment_id: recipe,
    )

    task = manager.prepare(recipe.environment_id)

    with pytest.raises(
        PluginEnvironmentProvisioningError,
        match='provisioning failed',
    ) as error:
        task.result(2)
    assert error.value.plugin == recipe.plugin
    assert error.value.environment == recipe.environment_id
    assert error.value.details == 'pixi exited with status 1'


def test_worker_failure_has_remote_details(
    environment_manager, monkeypatch
) -> None:
    manager, _ = environment_manager
    command = _command(
        'example-plugin.fail',
        _recipe(),
        target='worker:fail',
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )

    task = manager.execute(command.command_id, (), {})

    with pytest.raises(
        PluginWorkerError, match='remote worker failed'
    ) as error:
        task.result(2)
    assert error.value.command == command.command_id
    assert error.value.details == 'remote traceback: ValueError: bad input'
    assert error.value.failure is not None
    assert error.value.failure.category == 'remote_exception'
    assert error.value.failure.remote_exception_type == 'builtins.ValueError'


def test_release_plugin_and_shutdown_close_owned_resources(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    first = _command('plugin-a.compute', _recipe(plugin='plugin-a'))
    second = _command(
        'plugin-b.compute',
        _recipe(plugin='plugin-b', environment='plugin-b.worker'),
    )
    commands = {first.command_id: first, second.command_id: second}
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        commands.__getitem__,
    )
    manager.execute(first.command_id, (), {}).result()
    manager.execute(second.command_id, (), {}).result()

    manager.release_plugin(first.plugin)

    assert backend.started[0].closed
    assert not backend.started[1].closed
    manager.close()
    assert backend.started[1].closed
    assert backend.closed


def test_environment_inventory_tracks_persistent_and_worker_state(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    recipe = _recipe()
    command = _command('example-plugin.compute', recipe)
    manifest = _manifest_for_recipe(recipe)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (manifest,),
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )

    [missing] = manager.list_environments(recipe.plugin)
    assert missing.state is PluginEnvironmentState.MISSING
    assert missing.worker_state is PluginWorkerState.STOPPED
    assert missing.display_name == 'Example worker'
    assert missing.provision == 'on_demand'

    manager.prepare(recipe.environment_id).result()
    [ready] = manager.list_environments(recipe.plugin)
    assert ready.state is PluginEnvironmentState.READY
    assert ready.worker_state is PluginWorkerState.STOPPED

    manager.execute(command.command_id, (), {}).result()
    [running] = manager.list_environments(recipe.plugin)
    assert running.state is PluginEnvironmentState.READY
    assert running.worker_state is PluginWorkerState.RUNNING

    manager.stop_workers(recipe.plugin, recipe.environment_id).result()
    [stopped] = manager.list_environments(recipe.plugin)
    assert stopped.state is PluginEnvironmentState.READY
    assert stopped.worker_state is PluginWorkerState.STOPPED
    assert backend.started[-1].closed

    manager.execute(command.command_id, (), {}).result()
    assert len(backend.started) == 2


def test_changed_recipe_is_reported_stale(
    environment_manager, monkeypatch
) -> None:
    manager, _backend = environment_manager
    recipe = _recipe()
    selected_manifest = _manifest_for_recipe(recipe)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (selected_manifest,),
    )

    manager.prepare(recipe.environment_id).result()
    changed = _recipe(requirement='example-dependency==2')
    selected_manifest = _manifest_for_recipe(changed)

    [info] = manager.list_environments(recipe.plugin)
    assert info.state is PluginEnvironmentState.STALE


def test_ownership_survives_manager_restart_and_manifest_removal(
    tmp_path: Path, monkeypatch
) -> None:
    recipe = _recipe()
    manifest = _manifest_for_recipe(recipe)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (manifest,),
    )
    first_backend = _FakeBackend(tmp_path)
    first = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: first_backend,
    )
    first.prepare(recipe.environment_id).result()
    known = set(first_backend.known)
    first.close()

    second_backend = _FakeBackend(tmp_path)
    second_backend.known.update(known)
    second = PluginEnvironmentManager(
        root=tmp_path,
        backend_factory=lambda root: second_backend,
    )
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (),
    )
    try:
        [owned] = second.list_environments(recipe.plugin)
        assert owned.environment_id == recipe.environment_id
        assert owned.state is PluginEnvironmentState.READY

        second.remove_environments(recipe.plugin).result()
        assert second_backend.removed == list(known)
        assert second.list_environments(recipe.plugin) == ()
    finally:
        second.close()


def test_explicit_removal_failure_is_reported_and_retains_ownership(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    recipe = _recipe()
    manifest = _manifest_for_recipe(recipe)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (manifest,),
    )
    manager.prepare(recipe.environment_id).result()
    backend.fail_removal = True

    with pytest.raises(
        PluginEnvironmentProvisioningError, match='removal failed'
    ) as error:
        manager.remove_environments(recipe.plugin).result()

    assert error.value.environment == recipe.environment_id
    [info] = manager.list_environments(recipe.plugin)
    assert info.state is PluginEnvironmentState.READY


def test_lazy_preparation_is_visible_and_failure_is_retained(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    recipe = _recipe()
    command = _command('example-plugin.compute', recipe)
    manifest = _manifest_for_recipe(recipe)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (manifest,),
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        lambda command_id: command,
    )
    backend.block_preparation = True

    task = manager.execute(command.command_id, (), {})
    assert backend.preparation_started.wait(2)
    [preparing] = manager.list_environments(recipe.plugin)
    assert preparing.state is PluginEnvironmentState.PREPARING
    task.cancel()
    with pytest.raises(PluginTaskCanceledError):
        task.result(2)

    backend.block_preparation = False
    backend.fail_preparation = True
    backend.preparation_started.clear()
    failed_task = manager.execute(command.command_id, (), {})
    with pytest.raises(
        PluginEnvironmentProvisioningError, match='provisioning failed'
    ):
        failed_task.result(2)
    [failed] = manager.list_environments(recipe.plugin)
    assert failed.state is PluginEnvironmentState.FAILED
    assert failed.failure == 'provisioning failed'


def test_removing_one_environment_keeps_other_workers_running(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    first = _recipe(environment='example-plugin.first')
    second = _recipe(
        environment='example-plugin.second',
        requirement='example-dependency==2',
    )
    manifests = (
        _manifest_for_recipe(first, display_name='First'),
        _manifest_for_recipe(second, display_name='Second'),
    )
    commands = {
        'example-plugin.first': _command('example-plugin.first', first),
        'example-plugin.second': _command('example-plugin.second', second),
    }
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: manifests,
    )
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        commands.__getitem__,
    )
    manager.execute('example-plugin.first', (), {}).result()
    manager.execute('example-plugin.second', (), {}).result()

    manager.remove_environments(
        'example-plugin',
        (first.environment_id,),
    ).result()

    infos = {
        info.environment_id: info
        for info in manager.list_environments('example-plugin')
    }
    assert infos[first.environment_id].state is PluginEnvironmentState.MISSING
    assert (
        infos[first.environment_id].worker_state is PluginWorkerState.STOPPED
    )
    assert infos[second.environment_id].state is PluginEnvironmentState.READY
    assert (
        infos[second.environment_id].worker_state is PluginWorkerState.RUNNING
    )
    assert backend.started[0].closed
    assert not backend.started[1].closed


def test_invalid_worker_package_is_a_failed_inventory_item(
    environment_manager, tmp_path: Path, monkeypatch
) -> None:
    manager, _backend = environment_manager
    recipe = _recipe()
    manifest = _manifest_for_recipe(recipe)
    manifest._source_file = tmp_path / 'napari.yaml'
    worker = tmp_path / 'worker'
    worker.mkdir()
    (worker / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "example-worker"\n'
        'version = "1.0"\n'
        'dependencies = ["undeclared-dependency"]\n',
        encoding='utf-8',
    )
    manifest.contributions.environments[0].local_packages = [
        SimpleNamespace(path='worker')
    ]
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (manifest,),
    )

    [info] = manager.list_environments(recipe.plugin)
    assert info.state is PluginEnvironmentState.FAILED
    assert info.failure is not None
    assert 'runtime dependencies must be declared' in info.failure


def test_explicit_remove_cleans_failed_stale_generation(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    selected = _recipe()
    selected_manifest = _manifest_for_recipe(selected)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (selected_manifest,),
    )
    manager.prepare(selected.environment_id).result()
    old_name = next(iter(backend.known))

    selected = _recipe(requirement='example-dependency==2')
    selected_manifest = _manifest_for_recipe(selected)
    backend.removal_failures = 1
    manager.prepare(selected.environment_id).result()
    assert old_name in backend.known
    assert len(backend.known) == 2

    manager.remove_environments(selected.plugin).result()
    assert backend.known == set()
    assert manager.list_environments(selected.plugin)[0].state is (
        PluginEnvironmentState.MISSING
    )


def test_ownership_is_updated_before_old_generation_cleanup(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    selected = _recipe()
    selected_manifest = _manifest_for_recipe(selected)
    monkeypatch.setattr(
        manager_module,
        '_iter_manifests',
        lambda *, enabled_only=False: (selected_manifest,),
    )
    manager.prepare(selected.environment_id).result()
    old_name = next(iter(backend.known))
    record_ownership = manager._record_ownership

    selected = _recipe(requirement='example-dependency==2')
    selected_manifest = _manifest_for_recipe(selected)

    def fail_ownership(*args, **kwargs) -> None:
        raise OSError('ownership write failed')

    monkeypatch.setattr(manager, '_record_ownership', fail_ownership)
    with pytest.raises(
        PluginEnvironmentError,
        match='ownership write failed',
    ):
        manager.prepare(selected.environment_id).result()

    assert backend.removed == []
    assert old_name in backend.known
    assert len(backend.known) == 2

    monkeypatch.setattr(manager, '_record_ownership', record_ownership)
    manager.remove_environments(selected.plugin).result()
    assert backend.known == set()


def test_manifest_iteration_can_include_disabled_plugins(
    monkeypatch,
) -> None:
    calls: list[bool | None] = []

    def iter_manifests(*, disabled):
        calls.append(disabled)
        return ()

    monkeypatch.setattr(
        'npe2.plugin_manager.iter_manifests',
        iter_manifests,
    )

    assert manager_module._iter_manifests() == ()
    assert manager_module._iter_manifests(enabled_only=True) == ()
    assert calls == [None, False]


def test_embedded_worker_package_has_manifest_owned_dependencies(
    tmp_path: Path,
) -> None:
    worker = tmp_path / 'worker'
    worker.mkdir()
    pyproject = worker / 'pyproject.toml'
    pyproject.write_text(
        '[project]\nname = "example-worker"\nversion = "1.0"\n',
        encoding='utf-8',
    )

    recipe = manager_module._local_package_recipe(tmp_path, 'worker')
    assert recipe.path == worker

    pyproject.write_text(
        '[project]\n'
        'name = "example-worker"\n'
        'version = "1.0"\n'
        'dependencies = ["heavy-dependency"]\n',
        encoding='utf-8',
    )
    with pytest.raises(
        ValueError,
        match='runtime dependencies must be declared by the environment',
    ):
        manager_module._local_package_recipe(tmp_path, 'worker')


def test_embedded_worker_package_cannot_escape_plugin_package(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match='must remain inside'):
        manager_module._local_package_recipe(tmp_path, '../worker')


def test_worker_command_registry_uses_napari_proxy(
    npe2pm, monkeypatch
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def execute(
        command_id: str, *args: Any, **kwargs: Any
    ) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        call = (command_id, args, kwargs)
        calls.append(call)
        return call

    monkeypatch.setattr(
        'napari.plugins.environments.execute_worker_command',
        execute,
    )
    manifest = PluginManifest(
        name='isolated-example',
        schema_version='0.3.0',
        contributions={
            'environments': [
                {
                    'id': 'isolated-example.worker',
                    'display_name': 'Worker',
                    'python': '3.12',
                    'pypi': ['heavy-dependency==1'],
                }
            ],
            'commands': [
                {
                    'id': 'isolated-example.compute',
                    'title': 'Compute',
                    'python_name': 'unimportable_heavy_module:compute',
                    'environment': 'isolated-example.worker',
                }
            ],
        },
    )

    with npe2pm.tmp_plugin(manifest=manifest):
        _npe2._register_manifest_actions(manifest)
        result = npe2pm.commands.execute(
            'isolated-example.compute',
            args=(1,),
            kwargs={'value': 2},
        )

    assert result == (
        'isolated-example.compute',
        (1,),
        {'value': 2},
    )
    assert calls == [result]
    assert 'unimportable_heavy_module' not in __import__('sys').modules

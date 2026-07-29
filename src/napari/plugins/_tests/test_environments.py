from __future__ import annotations

import hashlib
import json
import threading
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
    PluginEnvironmentProvisioningError,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskPhase,
    PluginTaskState,
    PluginWorkerError,
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

    def remove_environment(self, physical_name: str) -> None:
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
        schema_version='0.4.0',
        contributions={
            'environments': [
                {
                    'id': 'isolated-example.worker',
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

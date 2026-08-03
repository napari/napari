from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from napari.plugins._environment_types import (
    BackendCanceled,
    BackendFailure,
    BackendUnavailable,
    LocalPackageRecipe,
)
from napari.plugins._tests.test_environments import _recipe
from napari.plugins._wetlands import (
    WetlandsBackend,
    WetlandsPool,
    _normalize_error,
)
from napari.plugins.environments import PluginTaskPhase

if TYPE_CHECKING:
    from pathlib import Path


class _OperationCanceled(RuntimeError):
    pass


class _ExecutionFailure(SimpleNamespace):
    @classmethod
    def environment(cls, message: str) -> _ExecutionFailure:
        """Match the Wetlands classmethod that collides with diagnostic data."""
        return cls(message=message)


@dataclass
class _LocalPackage:
    source: Path


class _EnvironmentSpec:
    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs
        self.recipe_hash = repr(sorted(kwargs.items()))


class _EnvironmentManager:
    def __init__(self, root: Path) -> None:
        self.root = root


def _wetlands_module(version: str = '2.2.0') -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        EnvironmentManager=_EnvironmentManager,
        EnvironmentSpec=_EnvironmentSpec,
        LocalPackage=_LocalPackage,
        OperationCanceled=_OperationCanceled,
    )


@pytest.mark.parametrize('version', ['1.1.1', '2.1.0', 'not-a-version'])
def test_backend_rejects_unsupported_wetlands_version(
    tmp_path: Path, monkeypatch, version: str
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module(version))

    with pytest.raises(
        BackendUnavailable, match=r'Wetlands 2\.2 or newer is required'
    ):
        WetlandsBackend(tmp_path)


def test_spec_maps_local_packages_and_lockfile(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module())
    backend = WetlandsBackend(tmp_path)
    recipe = _recipe()
    recipe = recipe.__class__(
        **{
            **recipe.__dict__,
            'local_packages': (LocalPackageRecipe(tmp_path / 'worker'),),
            'lockfile': b'lock contents',
        }
    )

    spec = backend._spec(recipe)

    assert spec.values['python'] == recipe.python
    assert spec.values['pypi'] == recipe.pypi
    assert spec.values['pixi_lock'] == b'lock contents'
    assert spec.values['local'] == (_LocalPackage(tmp_path / 'worker'),)


def test_backend_version_participates_in_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module('2.2.0'))
    first = WetlandsBackend(tmp_path).fingerprint(_recipe())
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module('2.3.0'))
    second = WetlandsBackend(tmp_path).fingerprint(_recipe())

    assert first != second


def test_backend_does_not_expose_sub_operation_completion() -> None:
    events = (
        SimpleNamespace(
            kind=SimpleNamespace(value='state'),
            stage=None,
            message='Operation started',
            current=None,
            maximum=None,
        ),
        SimpleNamespace(
            kind=SimpleNamespace(value='output'),
            stage=SimpleNamespace(value='install'),
            message='Installing packages',
            current=1,
            maximum=2,
        ),
        SimpleNamespace(
            kind=SimpleNamespace(value='state'),
            stage=None,
            message='Operation completed',
            current=None,
            maximum=None,
        ),
    )
    operation = SimpleNamespace(
        listen=lambda callback: [callback(event) for event in events]
    )
    received = []

    WetlandsBackend._listen_operation(
        operation,
        PluginTaskPhase.PROVISIONING,
        received.append,
    )

    assert [update.message for update in received] == [
        'Operation started',
        'install: Installing packages',
    ]


def test_pool_recognizes_public_cancellation_type() -> None:
    task = SimpleNamespace(
        cancel=lambda: True,
        listen=lambda callback: None,
        wait_for=lambda: (_ for _ in ()).throw(_OperationCanceled()),
    )
    worker_pool = SimpleNamespace(
        submit_import=lambda *args, **kwargs: task,
        close=lambda: None,
    )
    pool = WetlandsPool(worker_pool, _OperationCanceled)

    with pytest.raises(BackendCanceled):
        pool.execute(
            'worker:call',
            (),
            {},
            accepts_context=False,
            progress=lambda update: None,
            set_cancel_callback=lambda callback: None,
        )


def test_execution_failure_is_normalized_without_wetlands_types() -> None:
    remote = SimpleNamespace(
        qualified_name='example.RemoteError',
        message='bad input',
    )
    worker = SimpleNamespace(environment='plugin.worker', pid=44)
    failure = _ExecutionFailure(
        category=SimpleNamespace(value='remote_exception'),
        message='bad input',
        task_id='task-1',
        call_target='worker:call',
        traceback='remote traceback',
        remote_exception=remote,
        worker=worker,
        exit_code=None,
        signal=None,
        timeout=None,
        elapsed=0.25,
        serialization_context=None,
        summary=lambda: 'RemoteError: bad input',
    )
    error = RuntimeError('worker failed')
    error.failure = failure

    normalized = _normalize_error(error)

    assert str(normalized) == 'RemoteError: bad input'
    assert 'environment: <bound method' not in normalized.details
    assert 'worker_environment: plugin.worker' in normalized.details
    assert 'worker_pid: 44' in normalized.details
    assert normalized.diagnostics == {
        'category': 'remote_exception',
        'message': 'bad input',
        'target': 'worker:call',
        'traceback': 'remote traceback',
        'remote_exception_type': 'example.RemoteError',
        'remote_exception_message': 'bad input',
        'worker_environment': 'plugin.worker',
        'worker_pid': 44,
        'exit_code': None,
        'signal': None,
        'timeout': None,
        'elapsed': 0.25,
        'serialization_context': None,
    }


def test_pool_adds_missing_worker_diagnostics_from_registry() -> None:
    remote = SimpleNamespace(
        qualified_name='builtins.ModuleNotFoundError',
        message="No module named 'dependency'",
    )
    failure = _ExecutionFailure(
        category=SimpleNamespace(value='remote_exception'),
        message="No module named 'dependency'",
        task_id='task-1',
        call_target='worker:call',
        traceback='remote traceback',
        remote_exception=remote,
        worker=None,
        exit_code=None,
        signal=None,
        timeout=None,
        elapsed=None,
        serialization_context=None,
        summary=lambda: (
            "Remote ModuleNotFoundError: No module named 'dependency'"
        ),
    )
    execution_error = RuntimeError('worker failed')
    execution_error.failure = failure
    task = SimpleNamespace(
        cancel=lambda: True,
        listen=lambda callback: None,
        wait_for=lambda: (_ for _ in ()).throw(execution_error),
    )
    worker_pool = SimpleNamespace(
        submit_import=lambda *args, **kwargs: task,
        close=lambda: None,
    )
    pool = WetlandsPool(
        worker_pool,
        _OperationCanceled,
        environment_name='plugin-worker-a1b2',
        running_workers=lambda environment: (
            SimpleNamespace(
                environment=environment,
                process_id=1234,
            ),
        ),
    )

    with pytest.raises(BackendFailure) as exc_info:
        pool.execute(
            'worker:call',
            (),
            {},
            accepts_context=False,
            progress=lambda update: None,
            set_cancel_callback=lambda callback: None,
        )

    assert exc_info.value.diagnostics is not None
    assert (
        exc_info.value.diagnostics['worker_environment']
        == 'plugin-worker-a1b2'
    )
    assert exc_info.value.diagnostics['worker_pid'] == 1234
    assert 'environment: <bound method' not in exc_info.value.details
    assert 'worker_environment: plugin-worker-a1b2' in exc_info.value.details
    assert 'worker_pid: 1234' in exc_info.value.details

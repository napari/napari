from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from napari.plugins._environment_types import (
    BackendCanceled,
    BackendUnavailable,
    LocalPackageRecipe,
)
from napari.plugins._tests.test_environments import _recipe
from napari.plugins._wetlands import (
    WetlandsBackend,
    WetlandsPool,
    _normalize_error,
)

if TYPE_CHECKING:
    from pathlib import Path


class _OperationCanceled(RuntimeError):
    pass


@dataclass
class _LocalPackage:
    source: Path
    editable: bool
    extras: tuple[str, ...]


class _EnvironmentSpec:
    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs
        self.recipe_hash = repr(sorted(kwargs.items()))


class _EnvironmentManager:
    def __init__(self, root: Path) -> None:
        self.root = root


def _wetlands_module(version: str = '2.0.0') -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        EnvironmentManager=_EnvironmentManager,
        EnvironmentSpec=_EnvironmentSpec,
        LocalPackage=_LocalPackage,
        OperationCanceled=_OperationCanceled,
    )


def test_backend_rejects_unsupported_wetlands_version(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module('1.1.1'))

    with pytest.raises(BackendUnavailable, match='Wetlands 2 is required'):
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
            'local_packages': (
                LocalPackageRecipe(
                    tmp_path / 'worker',
                    editable=True,
                    extras=('gpu',),
                ),
            ),
            'lockfile': b'lock contents',
        }
    )

    spec = backend._spec(recipe)

    assert spec.values['python'] == recipe.python
    assert spec.values['pypi'] == recipe.pypi
    assert spec.values['pixi_lock'] == b'lock contents'
    assert spec.values['local'] == (
        _LocalPackage(tmp_path / 'worker', True, ('gpu',)),
    )


def test_backend_version_participates_in_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module('2.0.0'))
    first = WetlandsBackend(tmp_path).fingerprint(_recipe())
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module('2.1.0'))
    second = WetlandsBackend(tmp_path).fingerprint(_recipe())

    assert first != second


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
    failure = SimpleNamespace(
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

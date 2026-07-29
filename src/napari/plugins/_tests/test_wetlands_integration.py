"""Opt-in acceptance coverage for the real Wetlands 2 backend."""

from __future__ import annotations

import os
import sys
import threading
from importlib.metadata import version as distribution_version
from typing import TYPE_CHECKING

import numpy as np
import pytest

from napari.plugins import _environment_manager as manager_module
from napari.plugins._environment_manager import PluginEnvironmentManager
from napari.plugins._environment_types import (
    EnvironmentRecipe,
    LocalPackageRecipe,
    WorkerCommand,
)
from napari.plugins.environments import (
    PluginTaskCanceledError,
    PluginTaskPhase,
)

if TYPE_CHECKING:
    from pathlib import Path

_ENABLE_ENVIRONMENT_VARIABLE = 'NAPARI_RUN_WETLANDS_INTEGRATION'

pytestmark = pytest.mark.slow


def _require_wetlands_2() -> None:
    if os.environ.get(_ENABLE_ENVIRONMENT_VARIABLE) != '1':
        pytest.skip(
            f'set {_ENABLE_ENVIRONMENT_VARIABLE}=1 to provision real Pixi '
            'environments'
        )
    wetlands = pytest.importorskip('wetlands')
    version = getattr(wetlands, '__version__', '0')
    if version.partition('.')[0] != '2':
        pytest.skip(f'Wetlands 2 is required, found {version}')


def _create_worker_package(root: Path) -> Path:
    package = root / 'worker-package'
    module = package / 'src' / 'napari_wetlands_acceptance'
    module.mkdir(parents=True)
    (package / 'pyproject.toml').write_text(
        """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "napari-wetlands-acceptance"
version = "1.0.0"

[tool.hatch.build.targets.wheel]
packages = ["src/napari_wetlands_acceptance"]
""".lstrip(),
        encoding='utf-8',
    )
    (module / '__init__.py').write_text(
        """
import os
import time
from importlib.metadata import version

import numpy as np


def transform(payload, napari_context=None):
    napari_context.update("worker transform", current=1, maximum=2)
    result = {
        "pid": os.getpid(),
        "dependency_version": version("typing-extensions"),
        "payload": {
            "array": -payload["items"][0],
            "nested": payload["items"][1],
        },
    }
    napari_context.update("worker transform", current=2, maximum=2)
    return result


def wait_for_cancellation(napari_context=None):
    napari_context.update("waiting for cancellation")
    while not napari_context.cancel_requested:
        time.sleep(0.02)
    napari_context.cancel()
""".lstrip(),
        encoding='utf-8',
    )
    return package


def _recipe(
    package: Path,
    *,
    plugin: str,
    dependency_version: str,
) -> EnvironmentRecipe:
    return EnvironmentRecipe(
        plugin=plugin,
        plugin_version='1.0.0',
        environment_id=f'{plugin}.worker',
        python=f'{sys.version_info.major}.{sys.version_info.minor}.*',
        conda=('numpy', 'pip'),
        pypi=(f'typing-extensions=={dependency_version}',),
        channels=('conda-forge',),
        local_packages=(LocalPackageRecipe(package),),
        lockfile=None,
    )


def _worker_command(
    command_id: str,
    recipe: EnvironmentRecipe,
    target: str,
) -> WorkerCommand:
    return WorkerCommand(
        plugin=recipe.plugin,
        environment_id=recipe.environment_id,
        command_id=command_id,
        target=target,
        accepts_context=True,
        recipe=recipe,
    )


def test_real_wetlands_isolates_reuses_and_cancels_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the napari task API against two real Pixi environments."""

    _require_wetlands_2()
    package = _create_worker_package(tmp_path)
    old_recipe = _recipe(
        package,
        plugin='wetlands-old-dependency',
        dependency_version='4.8.0',
    )
    new_recipe = _recipe(
        package,
        plugin='wetlands-new-dependency',
        dependency_version='4.15.0',
    )
    commands = {
        'wetlands-old-dependency.transform': _worker_command(
            'wetlands-old-dependency.transform',
            old_recipe,
            'napari_wetlands_acceptance:transform',
        ),
        'wetlands-old-dependency.cancel': _worker_command(
            'wetlands-old-dependency.cancel',
            old_recipe,
            'napari_wetlands_acceptance:wait_for_cancellation',
        ),
        'wetlands-new-dependency.transform': _worker_command(
            'wetlands-new-dependency.transform',
            new_recipe,
            'napari_wetlands_acceptance:transform',
        ),
    }
    monkeypatch.setattr(
        manager_module,
        '_find_worker_command',
        commands.__getitem__,
    )
    monkeypatch.setattr(
        manager_module,
        '_owner_for_contribution',
        lambda contribution_id: None,
    )
    manager = PluginEnvironmentManager(root=tmp_path / 'managed')
    host_dependency_version = distribution_version('typing-extensions')
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    original = image.copy()
    payload = {
        'items': [
            image,
            {
                'labels': [1, 2, 3],
                'metadata': (None, True, 2.5, 'ordinary values'),
            },
        ]
    }

    try:
        old_progress = []
        old_task = manager.execute(
            'wetlands-old-dependency.transform',
            (payload,),
            {},
        )
        old_task.add_progress_callback(old_progress.append)
        old_result = old_task.result(900)
        repeated_result = manager.execute(
            'wetlands-old-dependency.transform',
            (payload,),
            {},
        ).result(120)
        new_result = manager.execute(
            'wetlands-new-dependency.transform',
            (payload,),
            {},
        ).result(900)

        assert old_result['pid'] != os.getpid()
        assert new_result['pid'] != os.getpid()
        assert old_result['pid'] != new_result['pid']
        assert repeated_result['pid'] == old_result['pid']
        assert old_result['dependency_version'] == '4.8.0'
        assert new_result['dependency_version'] == '4.15.0'
        assert (
            distribution_version('typing-extensions')
            == host_dependency_version
        )
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(old_result['payload']['array'], -image)
        assert old_result['payload']['nested'] == payload['items'][1]
        assert any(
            update.phase is PluginTaskPhase.EXECUTING
            and update.message == 'worker transform'
            and update.current == 2
            and update.total == 2
            for update in old_progress
        )

        cancellation_ready = threading.Event()
        canceled_task = manager.execute(
            'wetlands-old-dependency.cancel',
            (),
            {},
        )
        canceled_task.add_progress_callback(
            lambda update: (
                cancellation_ready.set()
                if update.message == 'waiting for cancellation'
                else None
            )
        )
        assert cancellation_ready.wait(60)
        assert canceled_task.cancel()
        with pytest.raises(PluginTaskCanceledError):
            canceled_task.result(60)

        recovered = manager.execute(
            'wetlands-old-dependency.transform',
            (payload,),
            {},
        ).result(120)
        assert recovered['dependency_version'] == '4.8.0'
    finally:
        manager.close()

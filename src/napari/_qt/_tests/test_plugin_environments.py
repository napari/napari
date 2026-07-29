from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from napari._qt import _plugin_environments as qt_environments
from napari.plugins.environments import (
    PluginEnvironmentProvisioningError,
    PluginTask,
    PluginTaskPhase,
)


class _FakeProgressBar:
    instances: list[_FakeProgressBar] = []

    def __init__(self, task: PluginTask[Any], description: str) -> None:
        self.task = task
        self.description = description
        self.total: int | None = None
        self.n: int | None = None
        self.closed = False
        self.events = SimpleNamespace(value=lambda **kwargs: None)
        self.instances.append(self)

    def set_description(self, description: str) -> None:
        self.description = description

    def close(self) -> None:
        self.closed = True


def test_progress_bar_cancel_forwards_to_task() -> None:
    task: PluginTask[None] = PluginTask()
    canceled = False

    def cancel_backend() -> None:
        nonlocal canceled
        canceled = True

    task._set_cancel_callback(cancel_backend)
    progress_bar = qt_environments._PluginTaskProgressBar(task, 'Preparing')

    progress_bar.cancel()

    assert canceled
    assert task.cancellation_requested


def test_monitor_presents_progress_and_structured_failure(
    qapp, monkeypatch
) -> None:
    _FakeProgressBar.instances.clear()
    received_errors: list[BaseException] = []
    monkeypatch.setattr(
        qt_environments,
        '_PluginTaskProgressBar',
        _FakeProgressBar,
    )
    monkeypatch.setattr(qt_environments, 'current_viewer', lambda: None)
    monkeypatch.setattr(
        qt_environments.notification_manager,
        'receive_error',
        lambda exc_type, error, traceback: received_errors.append(error),
    )
    task: PluginTask[None] = PluginTask()

    qt_environments._monitor_task(task)
    task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
    task._report_progress(
        PluginTaskPhase.PROVISIONING,
        'Installing dependencies',
        2,
        4,
    )
    error = PluginEnvironmentProvisioningError(
        'Environment installation failed',
        details='pixi exited with status 1',
    )
    task._set_error(error)

    progress_bar = _FakeProgressBar.instances[-1]
    assert progress_bar.description == 'Installing dependencies'
    assert progress_bar.n == 2
    assert progress_bar.total == 4
    assert progress_bar.closed
    assert received_errors == [error]


def test_qt_support_installation_is_idempotent(monkeypatch) -> None:
    dispatchers = []
    observers = []
    shutdown_callbacks = []
    app = SimpleNamespace(
        aboutToQuit=SimpleNamespace(
            connect=shutdown_callbacks.append,
        )
    )
    monkeypatch.setattr(qt_environments, '_installed', False)
    monkeypatch.setattr(
        qt_environments,
        '_set_task_dispatcher',
        dispatchers.append,
    )
    monkeypatch.setattr(
        qt_environments,
        '_add_task_observer',
        observers.append,
    )

    qt_environments.install_plugin_environment_qt_support(app)
    qt_environments.install_plugin_environment_qt_support(app)

    assert dispatchers == [qt_environments._dispatch_to_main_thread]
    assert observers == [qt_environments._monitor_task]
    assert shutdown_callbacks == [qt_environments._shutdown_with_notification]

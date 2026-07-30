from __future__ import annotations

from types import SimpleNamespace

from napari._qt import _plugin_environments as qt_environments
from napari.plugins.environments import (
    PluginEnvironmentProvisioningError,
    PluginTask,
)


def test_task_observer_only_presents_structured_failure(
    qapp, monkeypatch
) -> None:
    received_errors: list[BaseException] = []
    monkeypatch.setattr(
        qt_environments.notification_manager,
        'receive_error',
        lambda exc_type, error, traceback: received_errors.append(error),
    )
    task: PluginTask[None] = PluginTask()

    qt_environments._notify_task_failure(task)
    error = PluginEnvironmentProvisioningError(
        'Environment installation failed',
        details='pixi exited with status 1',
    )
    task._set_error(error)

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
    assert observers == [qt_environments._notify_task_failure]
    assert shutdown_callbacks == [qt_environments._shutdown_with_notification]

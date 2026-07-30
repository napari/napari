"""Qt presentation and lifecycle integration for plugin environment tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from superqt import ensure_main_thread

from napari.plugins.environments import (
    PluginTaskState,
    _add_task_observer,
    _set_task_dispatcher,
)
from napari.utils.notifications import notification_manager

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QApplication

    from napari.plugins.environments import PluginTask

_installed = False


def _dispatch_to_main_thread(callback: Callable[[], None]) -> None:
    ensure_main_thread(callback)()


@ensure_main_thread
def _notify_task_failure(task: PluginTask[Any]) -> None:
    """Report unhandled task failures without choosing a progress surface."""

    def receive_done(done_task: PluginTask[Any]) -> None:
        if (
            done_task.state is PluginTaskState.FAILED
            and done_task.error is not None
        ):
            error = done_task.error
            notification_manager.receive_error(
                type(error),
                error,
                error.__traceback__,
            )

    task.add_done_callback(receive_done)


def _shutdown_with_notification() -> None:
    from napari.plugins._environment_manager import (
        shutdown_plugin_environments,
    )

    try:
        shutdown_plugin_environments()
    except Exception as error:  # noqa: BLE001
        notification_manager.receive_error(
            type(error),
            error,
            error.__traceback__,
        )


def install_plugin_environment_qt_support(app: QApplication) -> None:
    """Install Qt dispatch, failure notification, and shutdown support once."""

    global _installed
    if _installed:
        return
    _installed = True
    _set_task_dispatcher(_dispatch_to_main_thread)
    _add_task_observer(_notify_task_failure)
    app.aboutToQuit.connect(_shutdown_with_notification)


__all__ = ('install_plugin_environment_qt_support',)

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
from napari.utils.progress import cancelable_progress
from napari.utils.task_status import Status
from napari.viewer import current_viewer

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QApplication

    from napari.plugins.environments import PluginTask, PluginTaskProgress

_installed = False


class _PluginTaskProgressBar(cancelable_progress):
    """A cancelable progress bar backed by a managed plugin task."""

    def __init__(self, task: PluginTask[Any], description: str) -> None:
        self._task = task
        super().__init__(total=0, desc=description)

    def cancel(self) -> None:
        super().cancel()
        self._task.cancel()


def _dispatch_to_main_thread(callback: Callable[[], None]) -> None:
    ensure_main_thread(callback)()


@ensure_main_thread
def _monitor_task(task: PluginTask[Any]) -> None:
    initial = task.progress
    description = (
        initial.message
        if initial is not None
        else 'Managed plugin task pending'
    )
    progress_bar = _PluginTaskProgressBar(task, description)

    viewer = current_viewer()
    window = viewer.window if viewer is not None else None
    task_status_id = (
        window._register_task_status(
            'plugin-environment',
            Status.PENDING,
            description,
            cancel_callback=task.cancel,
        )
        if window is not None
        else None
    )

    def update_status(status: Status, message: str) -> None:
        if window is not None and task_status_id is not None:
            window._update_task_status(
                task_status_id,
                status,
                description=message,
            )

    def receive_progress(update: PluginTaskProgress) -> None:
        progress_bar.set_description(update.message)
        if update.total is not None:
            progress_bar.total = update.total
        else:
            progress_bar.total = 0
        if update.current is not None:
            progress_bar.n = update.current
            progress_bar.events.value(value=progress_bar.n)
        else:
            progress_bar.n = 0
            progress_bar.events.value(value=progress_bar.n)
        update_status(Status.BUSY, update.message)

    def receive_done(done_task: PluginTask[Any]) -> None:
        if (
            done_task.state is PluginTaskState.FAILED
            and done_task.error is not None
        ):
            error = done_task.error
            update_status(Status.FAILED, str(error))
            notification_manager.receive_error(
                type(error),
                error,
                error.__traceback__,
            )
        elif done_task.state is PluginTaskState.CANCELED:
            update_status(Status.CANCELLED, 'Managed plugin task canceled')
        else:
            update_status(Status.COMPLETED, 'Managed plugin task completed')
        progress_bar.close()

    task.add_progress_callback(receive_progress)
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
    """Install the Qt dispatcher, presentation, and shutdown hook once."""

    global _installed
    if _installed:
        return
    _installed = True
    _set_task_dispatcher(_dispatch_to_main_thread)
    _add_task_observer(_monitor_task)
    app.aboutToQuit.connect(_shutdown_with_notification)


__all__ = ('install_plugin_environment_qt_support',)

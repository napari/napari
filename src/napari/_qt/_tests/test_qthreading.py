import warnings

import pytest

from napari._qt import qthreading
from napari.utils.notifications import notification_manager
from napari.utils.task_status import Status


def _warn():
    warnings.warn('from worker', stacklevel=1)


def _warn_gen():
    warnings.warn('from worker', stacklevel=1)
    yield


def _task_statuses(viewer) -> list[Status]:
    tasks = viewer.window._task_status_manager._tasks
    return [item.state()[2] for item in tasks.values()]


@pytest.mark.parametrize('func', [_warn, _warn_gen])
def test_worker_warning_is_relayed(qtbot, monkeypatch, func):
    """Warnings in a worker reach the notification manager.

    The thread pool deletes the worker as soon as it finishes, so the relay
    must still work once the worker object is gone.
    """
    received = []
    monkeypatch.setattr(
        notification_manager,
        'receive_warning',
        lambda *args: received.append(args),
    )
    worker = qthreading.create_worker(func)
    with qtbot.waitSignal(worker.finished):
        worker.start()
    qtbot.waitUntil(lambda: len(received) == 1)
    assert str(received[0][0]) == 'from worker'


def test_worker_success_sets_completed_task_status(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    worker = qthreading.create_worker(lambda: None)
    with qtbot.waitSignal(worker.finished):
        worker.start()
    qtbot.waitUntil(lambda: _task_statuses(viewer) == [Status.COMPLETED])


def test_worker_error_sets_failed_task_status(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    def _raise():
        raise ValueError('from worker')

    worker = qthreading.create_worker(_raise, _ignore_errors=True)
    with qtbot.waitSignal(worker.finished):
        worker.start()
    # `finished` comes after `errored` and must not overwrite the failure
    qtbot.waitUntil(lambda: _task_statuses(viewer) == [Status.FAILED])


def test_worker_quit_sets_cancelled_task_status(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    def _endless():
        while True:
            yield

    worker = qthreading.create_worker(_endless)
    with qtbot.waitSignal(worker.finished):
        worker.start()
        qtbot.wait(50)
        worker.quit()
    # `finished` comes after `aborted` and must not overwrite the cancel
    qtbot.waitUntil(lambda: _task_statuses(viewer) == [Status.CANCELLED])

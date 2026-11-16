from napari._qt import qthreading
from napari.utils.task_status import Status


def _task_statuses(viewer) -> list[Status]:
    tasks = viewer.window._task_status_manager._tasks
    return [item.state()[2] for item in tasks.values()]


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

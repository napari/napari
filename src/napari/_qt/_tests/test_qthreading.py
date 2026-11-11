import warnings

import pytest

from napari._qt import qthreading
from napari.utils.notifications import notification_manager


def _warn():
    warnings.warn('from worker', stacklevel=1)


def _warn_gen():
    warnings.warn('from worker', stacklevel=1)
    yield


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

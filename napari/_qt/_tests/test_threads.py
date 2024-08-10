from unittest.mock import MagicMock

import pytest

from napari._qt.threads.status_checker import StatusChecker
from napari.components import ViewerModel


@pytest.mark.usefixtures('qapp')
def test_create():
    StatusChecker(ViewerModel())


@pytest.mark.usefixtures('qapp')
def test_no_emmit_no_ref(monkeypatch):
    """Calling calculate_status should not emit after viewer is deleted."""
    model = ViewerModel()
    status_checker = StatusChecker(model)
    monkeypatch.setattr(
        status_checker,
        'status_and_tooltip_changed',
        MagicMock(side_effect=RuntimeError('Should not emit')),
    )
    del model
    status_checker.calculate_status()


def test_terminate_no_ref(monkeypatch):
    """Test that the thread terminates when the viewer is garbage collected."""
    model = ViewerModel()
    status_checker = StatusChecker(model)
    del model
    status_checker.run()
    assert not status_checker._terminate


def test_waiting_on_no_request(monkeypatch, qtbot):
    def _check_status(value):
        return value == ('Ready', '')

    model = ViewerModel()
    model.mouse_over_canvas = True
    status_checker = StatusChecker(model)
    status_checker.start()
    with qtbot.waitSignal(
        status_checker.status_and_tooltip_changed,
        timeout=1000,
        check_params_cb=_check_status,
    ):
        status_checker.trigger_status_update()
    status_checker.terminate()

    qtbot.wait_until(lambda: status_checker.isFinished())

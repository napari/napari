from unittest.mock import Mock, patch

import pytest
from qtpy.QtCore import QMutex, QThread, QTimer
from superqt.utils import qdebounced


class _TestThread(QThread):
    def __init__(self) -> None:
        super().__init__()
        self.mutex = QMutex()

    def run(self):
        self.mutex.lock()


@pytest.mark.disable_qthread_start
def test_disable_qthread(qapp):
    t = _TestThread()
    t.mutex.lock()
    t.start()
    assert not t.isRunning()
    t.mutex.unlock()


def test_qthread_running(qtbot):
    t = _TestThread()
    t.mutex.lock()
    t.start()
    assert t.isRunning()
    t.mutex.unlock()
    qtbot.waitUntil(t.isFinished, timeout=2000)


@pytest.mark.disable_qtimer_start
def test_disable_qtimer(qtbot):
    t = QTimer()
    t.setInterval(100)
    t.start()
    assert not t.isActive()

    # As qtbot uses a QTimer in waitUntil, we also test if timer disable does not break it
    th = _TestThread()
    th.mutex.lock()
    th.start()
    assert th.isRunning()
    th.mutex.unlock()
    qtbot.waitUntil(th.isFinished, timeout=2000)
    assert not th.isRunning()


@pytest.mark.usefixtures("disable_throttling")
@patch("qtpy.QtCore.QTimer.start")
def test_disable_throttle(start_mock):
    mock = Mock()

    @qdebounced(timeout=50)
    def f() -> str:
        mock()

    f()
    start_mock.assert_not_called()
    mock.assert_called_once()


@patch("qtpy.QtCore.QTimer.start")
@patch("qtpy.QtCore.QTimer.isActive", return_value=True)
def test_lack_disable_throttle(_active_mock, start_mock, monkeypatch):
    """This is test showing that if we do not use disable_throttling then timer is started"""
    mock = Mock()

    @qdebounced(timeout=50)
    def f() -> str:
        mock()

    f()
    start_mock.assert_called_once()
    mock.assert_not_called()

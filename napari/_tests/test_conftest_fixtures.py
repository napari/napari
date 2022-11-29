import pytest
from qtpy.QtCore import QMutex, QThread, QTimer


class _TestThread(QThread):
    def __init__(self):
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

    th = _TestThread()
    th.mutex.lock()
    th.start()
    assert th.isRunning()
    th.mutex.unlock()
    qtbot.waitUntil(th.isFinished, timeout=2000)
    assert not th.isRunning()

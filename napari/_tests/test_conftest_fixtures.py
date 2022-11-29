import time

import pytest
from qtpy.QtCore import QThread, QTimer


class _TestThread(QThread):
    def run(self):
        time.sleep(1)


@pytest.mark.disable_qthread_start
def test_disable_qthread(qapp):
    t = _TestThread()
    t.start()
    assert not t.isRunning()


def test_qthread_running(qtbot):
    t = _TestThread()
    t.start()
    assert t.isRunning()
    qtbot.waitUntil(t.isFinished, timeout=2000)


@pytest.mark.disable_qtimer_start
def test_disable_qtimer(qtbot):
    t = QTimer()
    t.setInterval(100)
    t.start()
    assert not t.isActive()

    th = _TestThread()
    th.start()
    assert th.isRunning()
    qtbot.waitUntil(th.isFinished, timeout=2000)
    assert not th.isRunning()

from qtpy.QtCore import QObject, Signal

from ..util import blocked_signals


class Emitter(QObject):
    test_signal = Signal()

    def go(self):
        self.test_signal.emit()


def test_signal_blocker(qtbot):
    """make sure context manager signal blocker works"""

    obj = Emitter()

    # make sure signal works
    with qtbot.waitSignal(obj.test_signal):
        obj.go()

    # make sure blocker works
    def err():
        raise AssertionError('a signal was emitted')

    obj.test_signal.connect(err)
    with blocked_signals(obj):
        obj.go()
        qtbot.wait(750)

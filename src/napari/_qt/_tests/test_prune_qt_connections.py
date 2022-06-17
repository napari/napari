from unittest.mock import Mock

from qtpy.QtWidgets import QSpinBox

from napari.utils.events import EmitterGroup


def test_prune_dead_qt(qtbot):
    qtcalls = 0

    class W(QSpinBox):
        def _set(self, event):
            self.setValue(event.value)
            nonlocal qtcalls
            qtcalls += 1

    wdg = W()

    mock = Mock()
    group = EmitterGroup(None, False, boom=None)
    group.boom.connect(mock)
    group.boom.connect(wdg._set)
    assert len(group.boom.callbacks) == 2

    group.boom(value=1)
    assert qtcalls == 1
    mock.assert_called_once()
    mock.reset_mock()

    with qtbot.waitSignal(wdg.destroyed):
        wdg.close()
        wdg.deleteLater()

    group.boom(value=1)
    mock.assert_called_once()
    assert len(group.boom.callbacks) == 1  # we've lost the qt connection
    assert qtcalls == 1  # qwidget didn't get called again

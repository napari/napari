import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari._qt.widgets.qt_large_int_spinbox import QtLargeIntSpinBox


def test_large_spinbox(qtbot):
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)

    for e in range(2, 100, 2):
        sb.setMaximum(10 ** e + 2)
        with qtbot.waitSignal(sb.valueChanged) as sgnl:
            sb.setValue(10 ** e)
        assert sgnl.args == [10 ** e]
        assert sb.value() == 10 ** e

        sb.setMinimum(-(10 ** e) - 2)

        with qtbot.waitSignal(sb.valueChanged) as sgnl:
            sb.setValue(-(10 ** e))
        assert sgnl.args == [-(10 ** e)]
        assert sb.value() == -(10 ** e)


@pytest.mark.parametrize(
    "dtype",
    [int, 'uint8', np.uint8, 'int8', 'uint16', 'int16', 'uint32', 'int32'],
)
def test_clamp_dtype(qtbot, dtype):
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)
    sb.set_dtype(dtype)
    iinfo = np.iinfo(dtype)
    assert sb.minimum() == iinfo.min
    assert sb.maximum() == iinfo.max


def test_large_spinbox_type(qtbot):
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)

    assert isinstance(sb.value(), int)

    sb.setValue(1.1)
    assert isinstance(sb.value(), int)
    assert sb.value() == 1

    sb.setValue(1.9)
    assert isinstance(sb.value(), int)
    assert sb.value() == 1


def test_large_spinbox_signals(qtbot):
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)

    with qtbot.waitSignal(sb.valueChanged) as sgnl:
        sb.setValue(200)
    assert sgnl.args == [200]

    with qtbot.waitSignal(sb.textChanged) as sgnl:
        sb.setValue(240)
    assert sgnl.args == ['240']


def test_keyboard_tracking(qtbot):
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)
    assert sb.value() == 0
    sb.setKeyboardTracking(False)
    with qtbot.assertNotEmitted(sb.valueChanged):
        sb.lineEdit().setText('20')
    assert sb.lineEdit().text() == '20'
    assert sb.value() == 0
    assert sb._pending_emit is True

    with qtbot.waitSignal(sb.valueChanged) as sgnl:
        qtbot.keyPress(sb, Qt.Key_Enter)
    assert sgnl.args == [20]
    assert sb._pending_emit is False

    sb.setKeyboardTracking(True)
    with qtbot.waitSignal(sb.valueChanged) as sgnl:
        sb.lineEdit().setText('25')
    assert sb._pending_emit is False
    assert sgnl.args == [25]


def test_in_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer._new_labels()

    layer_controls = viewer.window.qt_viewer.dockLayerControls.widget()
    label_spinbox = layer_controls.widgets[viewer.layers[0]].selectionSpinBox
    label_spinbox.setValue(5)

    label_spinbox.setMaximum(2 ** 64)
    label_spinbox.setValue(2 ** 64 - 1)
    assert label_spinbox.value() == 2 ** 64 - 1

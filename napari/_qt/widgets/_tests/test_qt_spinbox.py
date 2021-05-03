from napari._qt.widgets.qt_large_spinbox import QtLargeIntSpinBox


def test_large_spinbox(qtbot):
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)
    sb = QtLargeIntSpinBox()
    qtbot.addWidget(sb)

    for e in range(2, 100, 2):
        sb.setMaximum(10 ** e + 2)
        sb.setValue(10 ** e)
        assert sb.value() == 10 ** e

        sb.setMinimum(-(10 ** e) - 2)
        sb.setValue(-(10 ** e))
        assert sb.value() == -(10 ** e)


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

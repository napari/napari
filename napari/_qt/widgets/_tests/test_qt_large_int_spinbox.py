from ..qt_large_int_spinbox import QtLargeIntSpinBox


def test_large_ints(qtbot):
    spinbox = QtLargeIntSpinBox()

    spinbox.setValue(1e34)
    assert spinbox.value() == spinbox.maximum()

    spinbox.setValue(1e29)
    assert spinbox.value() == spinbox.maximum()

    spinbox.setValue(1e13)
    assert spinbox.value() == int(1e13)

    spinbox.setValue(2 ** 64 - 1)
    assert spinbox.value() == spinbox.maximum()


def test_digitize_value(qtbot):
    spinbox = QtLargeIntSpinBox()

    spinbox.setValue(1.1)
    assert spinbox.value() == 1

    spinbox.setValue(1.9)
    assert spinbox.value() == 1


def test_event_emits_int(qtbot):
    def fn(value):
        assert isinstance(value, int)

    spinbox = QtLargeIntSpinBox()
    spinbox.valueChanged.connect(fn)
    spinbox.setValue(1)

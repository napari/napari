import numpy as np
import pytest

from ..qt_large_int_spinbox import QtLargeIntSpinBox


@pytest.mark.parametrize(
    ["dtype"],
    [
        (np.uint8,),
        (np.int8,),
        (np.uint16,),
        (np.int16,),
        (np.uint32,),
        (np.int32,),
        ("uint8",),
        (np.dtype("uint8"),),
    ],
)
def test_clamp_dtype(qtbot, dtype):
    spinbox = QtLargeIntSpinBox(dtype)
    iinfo = np.iinfo(dtype)
    assert spinbox.minimum() == iinfo.min
    assert spinbox.maximum() == iinfo.max


def test_clamp_float(qtbot):
    dtype = np.uint64
    spinbox = QtLargeIntSpinBox(dtype)
    iinfo = np.iinfo(dtype)
    assert spinbox.minimum() == iinfo.min
    assert spinbox.maximum() < iinfo.max


def test_digitize_value(qtbot):
    dtype = np.dtype(np.uint32)
    spinbox = QtLargeIntSpinBox(dtype)

    assert isinstance(spinbox.value(), dtype.type)

    spinbox.setValue(1.1)
    assert isinstance(spinbox.value(), dtype.type)
    assert spinbox.value() == 1

    spinbox.setValue(1.9)
    assert isinstance(spinbox.value(), dtype.type)
    assert spinbox.value() == 1


def test_event_emits_int(qtbot):
    dtype = np.dtype(np.int32)

    def fn(value):
        assert isinstance(value, int)

    spinbox = QtLargeIntSpinBox(dtype)
    spinbox.valueChanged.connect(fn)
    spinbox.setValue(1)

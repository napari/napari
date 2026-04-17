import numpy as np

from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari.layers import Vectors

_VECTORS = np.zeros((2, 2, 2))


def test_out_of_slice_display_checkbox(qtbot):
    layer = Vectors(_VECTORS)
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl._out_slice_checkbox_control.out_of_slice_checkbox.setChecked(True)
    assert layer.out_of_slice_display
    qtctrl._out_slice_checkbox_control.out_of_slice_checkbox.setChecked(False)
    assert not layer.out_of_slice_display

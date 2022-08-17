import numpy as np

from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari.layers import Points


def test_out_of_slice_display_checkbox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Points(np.random.rand(10, 2))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.outOfSliceCheckBox

    assert layer.out_of_slice_display is False
    combo.setChecked(True)
    assert layer.out_of_slice_display is True


def test_current_size_display_in_range(qtbot):
    """Changing the model attribute should update the view"""
    layer = Points(np.random.rand(10, 2))
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    slider = qtctrl.sizeSlider
    slider.setValue(10)

    # Initial values
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 10

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 5
    layer.events.size()
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 5

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 100
    layer.events.size()
    assert slider.maximum() == 100
    assert slider.minimum() == 1
    assert slider.value() == 100

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = 200
    layer.events.size()
    assert slider.maximum() == 201
    assert slider.minimum() == 1
    assert slider.value() == 200

    # Size event needs to be triggered manually, because no points are selected.
    layer.current_size = -1000
    layer.events.size()
    assert slider.maximum() == 201
    assert slider.minimum() == -1001
    assert slider.value() == -1000

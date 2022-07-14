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


def test_select_all_selects_value_change(qtbot):
    layer = Points(
        np.random.rand(10, 2), size=5, edge_color='red', face_color='red'
    )
    qtctrl = QtPointsControls(layer)
    qtbot.addWidget(qtctrl)
    size_ctrl = qtctrl.sizeSlider
    face_ctrl = qtctrl.faceColorEdit
    edge_ctrl = qtctrl.edgeColorEdit
    select_all_ctrl = qtctrl.selectAllCheckBox

    np.testing.assert_array_equal(layer.size, [[5] * 2] * 10)
    assert layer.current_size == 5
    np.testing.assert_almost_equal(layer.edge_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_edge_color == 'red'
    np.testing.assert_almost_equal(layer.face_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_face_color == 'red'

    size_ctrl.setValue(10)
    face_ctrl.setColor('green')
    edge_ctrl.setColor('green')

    np.testing.assert_array_equal(layer.size, [[5] * 2] * 10)
    assert layer.current_size == 10
    np.testing.assert_almost_equal(layer.face_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_face_color == 'green'
    np.testing.assert_almost_equal(layer.edge_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_edge_color == 'green'

    select_all_ctrl.setChecked(True)

    size_ctrl.setValue(15)
    face_ctrl.setColor('blue')
    edge_ctrl.setColor('blue')

    np.testing.assert_array_equal(layer.size, [[15] * 2] * 10)
    assert layer.current_size == 15
    np.testing.assert_almost_equal(layer.face_color, [[0, 0, 1, 1]] * 10)
    assert layer.current_face_color == 'blue'
    np.testing.assert_almost_equal(layer.edge_color, [[0, 0, 1, 1]] * 10)
    assert layer.current_edge_color == 'blue'

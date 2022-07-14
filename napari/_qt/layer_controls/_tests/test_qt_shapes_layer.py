import numpy as np

from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari.layers import Shapes
from napari.utils.colormaps.standardize_color import transform_color

_SHAPES = np.random.random((10, 4, 2))


def test_shape_controls_face_color(qtbot):
    """Check updating of face color updates QtShapesControls."""
    layer = Shapes(_SHAPES)
    qtctrl = QtShapesControls(layer)
    qtbot.addWidget(qtctrl)
    target_color = transform_color(layer.current_face_color)[0]
    np.testing.assert_almost_equal(qtctrl.faceColorEdit.color, target_color)

    # Update current face color
    layer.current_face_color = 'red'
    target_color = transform_color(layer.current_face_color)[0]
    np.testing.assert_almost_equal(qtctrl.faceColorEdit.color, target_color)


def test_shape_controls_edge_color(qtbot):
    """Check updating of edge color updates QtShapesControls."""
    layer = Shapes(_SHAPES)
    qtctrl = QtShapesControls(layer)
    qtbot.addWidget(qtctrl)
    target_color = transform_color(layer.current_edge_color)[0]
    np.testing.assert_almost_equal(qtctrl.edgeColorEdit.color, target_color)

    # Update current edge color
    layer.current_edge_color = 'red'
    target_color = transform_color(layer.current_edge_color)[0]
    np.testing.assert_almost_equal(qtctrl.edgeColorEdit.color, target_color)


def test_select_all_selects_value_change(qtbot):
    layer = Shapes(
        _SHAPES,
        edge_width=5,
        edge_color='red',
        face_color='red',
    )

    qtctrl = QtShapesControls(layer)
    qtbot.addWidget(qtctrl)
    size_ctrl = qtctrl.widthSlider
    face_ctrl = qtctrl.faceColorEdit
    edge_ctrl = qtctrl.edgeColorEdit
    select_all_ctrl = qtctrl.selectAllCheckBox

    np.testing.assert_array_equal(layer.edge_width, [5] * 10)
    assert layer.current_edge_width == 5
    np.testing.assert_almost_equal(layer.edge_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_edge_color == 'red'
    np.testing.assert_almost_equal(layer.face_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_face_color == 'red'

    size_ctrl.setValue(10)
    face_ctrl.setColor('green')
    edge_ctrl.setColor('green')

    np.testing.assert_array_equal(layer.edge_width, [5] * 10)
    assert layer.current_edge_width == 10
    np.testing.assert_almost_equal(layer.face_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_face_color == 'green'
    np.testing.assert_almost_equal(layer.edge_color, [[1, 0, 0, 1]] * 10)
    assert layer.current_edge_color == 'green'

    select_all_ctrl.setChecked(True)

    size_ctrl.setValue(15)
    face_ctrl.setColor('blue')
    edge_ctrl.setColor('blue')

    np.testing.assert_array_equal(layer.edge_width, [15] * 10)
    assert layer.current_edge_width == 15
    np.testing.assert_almost_equal(layer.face_color, [[0, 0, 1, 1]] * 10)
    assert layer.current_face_color == 'blue'
    np.testing.assert_almost_equal(layer.edge_color, [[0, 0, 1, 1]] * 10)
    assert layer.current_edge_color == 'blue'

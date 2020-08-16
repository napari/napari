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

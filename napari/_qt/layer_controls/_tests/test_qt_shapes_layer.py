import numpy as np
from qtpy.QtWidgets import QAbstractButton

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


def test_set_visible_or_editable_enables_edit_buttons(qtbot):
    """See https://github.com/napari/napari/issues/1346"""
    layer = Shapes(np.empty((0, 2, 4)))
    qtctrl = QtShapesControls(layer)
    qtbot.addWidget(qtctrl)
    assert all(map(QAbstractButton.isEnabled, qtctrl._EDIT_BUTTONS))

    layer.editable = False
    assert not any(map(QAbstractButton.isEnabled, qtctrl._EDIT_BUTTONS))

    layer.visible = False
    assert not any(map(QAbstractButton.isEnabled, qtctrl._EDIT_BUTTONS))

    layer.visible = True
    assert not any(map(QAbstractButton.isEnabled, qtctrl._EDIT_BUTTONS))

    layer.editable = True
    assert all(map(QAbstractButton.isEnabled, qtctrl._EDIT_BUTTONS))

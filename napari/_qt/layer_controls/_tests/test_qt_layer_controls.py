from collections import namedtuple

import numpy as np
import pytest

from napari._qt.layer_controls.qt_layer_controls_container import (
    create_qt_layer_controls,
    layer_to_controls,
)
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari.layers import Points, Shapes

LayerTypeWithData = namedtuple('LayerTypeWithData', ['type', 'data'])
_POINTS = LayerTypeWithData(type=Points, data=np.random.random((5, 2)))
_SHAPES = LayerTypeWithData(type=Shapes, data=np.random.random((10, 4, 2)))
_LINES_DATA = np.random.random((6, 2, 2))


def test_create_shape(qtbot):
    shapes = _SHAPES.type(_SHAPES.data)

    ctrl = create_qt_layer_controls(shapes)
    qtbot.addWidget(ctrl)

    assert isinstance(ctrl, QtShapesControls)


def test_unknown_raises(qtbot):
    class Test:
        """Unmatched class"""

    with pytest.raises(TypeError):
        create_qt_layer_controls(Test())


def test_inheritance(qtbot):
    class QtLinesControls(QtShapesControls):
        """Yes I'm the same"""

    class Lines(Shapes):
        """Here too"""

    lines = Lines(_LINES_DATA)
    layer_to_controls[Lines] = QtLinesControls
    ctrl = create_qt_layer_controls(lines)
    qtbot.addWidget(ctrl)
    assert isinstance(ctrl, QtLinesControls)


@pytest.mark.parametrize('layer_type_with_data', [_POINTS, _SHAPES])
def test_text_set_visible_updates_checkbox(qtbot, layer_type_with_data):
    text = {
        'text': 'test',
        'visible': True,
    }
    layer = layer_type_with_data.type(layer_type_with_data.data, text=text)
    ctrl = create_qt_layer_controls(layer)
    qtbot.addWidget(ctrl)
    assert ctrl.textDispCheckBox.isChecked()

    layer.text.visible = False

    assert not ctrl.textDispCheckBox.isChecked()


@pytest.mark.parametrize('layer_type_with_data', [_POINTS, _SHAPES])
def test_set_text_then_set_visible_updates_checkbox(
    qtbot, layer_type_with_data
):
    layer = layer_type_with_data.type(layer_type_with_data.data)
    ctrl = create_qt_layer_controls(layer)
    qtbot.addWidget(ctrl)
    layer.text = {
        'text': 'another_test',
        'visible': False,
    }
    assert not ctrl.textDispCheckBox.isChecked()

    layer.text.visible = True

    assert ctrl.textDispCheckBox.isChecked()

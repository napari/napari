import numpy as np
import pytest

from napari._qt.layer_controls.qt_layer_controls_container import (
    create_qt_layer_controls,
    layer_to_controls,
)
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari.layers import Shapes

_SHAPES = np.random.random((10, 4, 2))
_LINES = np.random.random((6, 2, 2))


def test_create_shape(qtbot):
    shapes = Shapes(_SHAPES)

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

    lines = Lines(_LINES)
    layer_to_controls[Lines] = QtLinesControls
    ctrl = create_qt_layer_controls(lines)
    qtbot.addWidget(ctrl)
    assert isinstance(ctrl, QtLinesControls)

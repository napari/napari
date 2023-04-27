from collections import namedtuple

import numpy as np
import pytest
from qtpy.QtWidgets import QAbstractButton

from napari._qt.layer_controls.qt_layer_controls_container import (
    QtLayerControlsContainer,
    create_qt_layer_controls,
    layer_to_controls,
)
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari.components import ViewerModel
from napari.layers import Labels, Points, Shapes

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
        'string': {'constant': 'test'},
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
        'string': {'constant': 'another_test'},
        'visible': False,
    }
    assert not ctrl.textDispCheckBox.isChecked()

    layer.text.visible = True

    assert ctrl.textDispCheckBox.isChecked()


@pytest.mark.parametrize(('ndim', 'editable_after'), ((2, False), (3, True)))
def test_set_3d_display_with_points(qtbot, ndim, editable_after):
    """Interactivity only works for 2D points layers rendered in 2D and not
    in 3D. Verify that layer.editable is set appropriately upon switching to
    3D rendering mode.

    See: https://github.com/napari/napari/pull/4184
    """
    viewer = ViewerModel()
    container = QtLayerControlsContainer(viewer)
    qtbot.addWidget(container)
    layer = viewer.add_points(np.zeros((0, ndim)), ndim=ndim)
    assert viewer.dims.ndisplay == 2
    assert layer.editable

    viewer.dims.ndisplay = 3

    assert layer.editable == editable_after


def test_set_3d_display_with_shapes(qtbot):
    """Interactivity only works for shapes layers rendered in 2D and not
    in 3D. Verify that layer.editable is set appropriately upon switching to
    3D rendering mode.

    See: https://github.com/napari/napari/pull/4184
    """
    viewer = ViewerModel()
    container = QtLayerControlsContainer(viewer)
    qtbot.addWidget(container)
    layer = viewer.add_shapes(np.zeros((0, 2, 4)))
    assert viewer.dims.ndisplay == 2
    assert layer.editable

    viewer.dims.ndisplay = 3

    assert not layer.editable


# The following tests handle changes to the layer's visible and
# editable state for layer control types that have controls to edit
# the layer. For more context see:
# https://github.com/napari/napari/issues/1346


@pytest.fixture(
    params=(
        (Labels, np.zeros((3, 4), dtype=int)),
        (Points, np.empty((0, 2))),
        (Shapes, np.empty((0, 2, 4))),
    )
)
def editable_layer(request):
    LayerType, data = request.param
    return LayerType(data)


def test_make_visible_when_editable_enables_edit_buttons(
    qtbot, editable_layer
):
    editable_layer.editable = True
    editable_layer.visible = False
    controls = make_layer_controls(qtbot, editable_layer)
    assert_no_edit_buttons_enabled(controls)

    editable_layer.visible = True

    assert_all_edit_buttons_enabled(controls)


def test_make_not_visible_when_editable_disables_edit_buttons(
    qtbot, editable_layer
):
    editable_layer.editable = True
    editable_layer.visible = True
    controls = make_layer_controls(qtbot, editable_layer)
    assert_all_edit_buttons_enabled(controls)

    editable_layer.visible = False

    assert_no_edit_buttons_enabled(controls)


def test_make_editable_when_visible_enables_edit_buttons(
    qtbot, editable_layer
):
    editable_layer.editable = False
    editable_layer.visible = True
    controls = make_layer_controls(qtbot, editable_layer)
    assert_no_edit_buttons_enabled(controls)

    editable_layer.editable = True

    assert_all_edit_buttons_enabled(controls)


def test_make_not_editable_when_visible_disables_edit_buttons(
    qtbot, editable_layer
):
    editable_layer.editable = True
    editable_layer.visible = True
    controls = make_layer_controls(qtbot, editable_layer)
    assert_all_edit_buttons_enabled(controls)

    editable_layer.editable = False

    assert_no_edit_buttons_enabled(controls)


def make_layer_controls(qtbot, layer):
    QtLayerControlsType = layer_to_controls[type(layer)]
    controls = QtLayerControlsType(layer)
    qtbot.addWidget(controls)
    return controls


def assert_all_edit_buttons_enabled(controls) -> None:
    assert all(map(QAbstractButton.isEnabled, controls._EDIT_BUTTONS))


def assert_no_edit_buttons_enabled(controls) -> None:
    assert not any(map(QAbstractButton.isEnabled, controls._EDIT_BUTTONS))

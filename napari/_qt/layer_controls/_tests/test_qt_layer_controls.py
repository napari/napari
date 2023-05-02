import os
import random
import sys
from collections import namedtuple

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QRadioButton,
)

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.layer_controls.qt_layer_controls_container import (
    QtLayerControlsContainer,
    create_qt_layer_controls,
    layer_to_controls,
)
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.components import ViewerModel
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)

np.random.seed(0)
LayerTypeWithData = namedtuple(
    'LayerTypeWithData',
    ['type', 'data', 'color', 'properties', 'expected_isinstance'],
)
_IMAGE = LayerTypeWithData(
    type=Image,
    data=np.random.rand(8, 8),
    color=None,
    properties=None,
    expected_isinstance=QtImageControls,
)
_LABELS_WITH_COLOR = LayerTypeWithData(
    type=Labels,
    data=np.random.randint(5, size=(10, 15)),
    color={1: 'white', 2: 'blue', 3: 'green', 4: 'red', 5: 'yellow'},
    properties=None,
    expected_isinstance=QtLabelsControls,
)
_LABELS = LayerTypeWithData(
    type=Labels,
    data=np.random.randint(5, size=(10, 15)),
    color=None,
    properties=None,
    expected_isinstance=QtLabelsControls,
)
_POINTS = LayerTypeWithData(
    type=Points,
    data=np.random.random((5, 2)),
    color=None,
    properties=None,
    expected_isinstance=QtPointsControls,
)
_SHAPES = LayerTypeWithData(
    type=Shapes,
    data=np.random.random((10, 4, 2)),
    color=None,
    properties=None,
    expected_isinstance=QtShapesControls,
)
_SURFACE = LayerTypeWithData(
    type=Surface,
    data=(
        np.random.random((10, 2)),
        np.random.randint(10, size=(6, 3)),
        np.random.random(10),
    ),
    color=None,
    properties=None,
    expected_isinstance=QtSurfaceControls,
)
_TRACKS = LayerTypeWithData(
    type=Tracks,
    data=np.zeros((2, 4)),
    color=None,
    properties={
        'track_id': [0, 0],
        'time': [0, 0],
        'speed': [50, 30],
    },
    expected_isinstance=QtTracksControls,
)
_VECTORS = LayerTypeWithData(
    type=Vectors,
    data=np.zeros((2, 2, 2)),
    color=None,
    properties=None,
    expected_isinstance=QtVectorsControls,
)
_LINES_DATA = np.random.random((6, 2, 2))


@pytest.fixture
def create_layer_controls(qtbot):
    def _create_layer_controls(layer_type_with_data):
        if layer_type_with_data.color:
            layer = layer_type_with_data.type(
                layer_type_with_data.data, color=layer_type_with_data.color
            )
        elif layer_type_with_data.properties:
            layer = layer_type_with_data.type(
                layer_type_with_data.data,
                properties=layer_type_with_data.properties,
            )
        else:
            layer = layer_type_with_data.type(layer_type_with_data.data)

        ctrl = create_qt_layer_controls(layer)
        qtbot.addWidget(ctrl)

        return ctrl

    return _create_layer_controls


@pytest.mark.parametrize(
    'layer_type_with_data',
    [
        _LABELS_WITH_COLOR,
        _LABELS,
        _IMAGE,
        _LABELS,
        _POINTS,
        _SHAPES,
        _SURFACE,
        _TRACKS,
        _VECTORS,
    ],
)
@pytest.mark.qt_no_exception_capture
@pytest.mark.skipif(os.environ.get("MIN_REQ", "0") == "1", reason="min req")
def test_create_layer_controls(
    qtbot, create_layer_controls, layer_type_with_data, capsys
):
    # create layer controls widget
    ctrl = create_layer_controls(layer_type_with_data)

    # check create widget corresponds to the expected class for each type of layer
    assert isinstance(ctrl, layer_type_with_data.expected_isinstance)

    # check QComboBox by changing current index
    for qcombobox in ctrl.findChildren(QComboBox):
        if qcombobox.isVisible():
            qcombobox_count = qcombobox.count()
            qcombobox_initial_idx = qcombobox.currentIndex()
            if qcombobox_count:
                qcombobox.setCurrentIndex(0)
            for idx in range(qcombobox_count):
                previous_qcombobox_text = qcombobox.currentText()
                qcombobox.setCurrentIndex(idx)
                # If a value for the QComboBox is an invalid selection check if
                # it fallbacks to the previous value
                captured = capsys.readouterr()
                if captured.err:
                    assert qcombobox.currentText() == previous_qcombobox_text
            qcombobox.setCurrentIndex(qcombobox_initial_idx)

    # check QAbstractSpinBox by changing value with `setValue` from minimum value to maximum
    for qspinbox in ctrl.findChildren(QAbstractSpinBox):
        qspinbox_initial_value = qspinbox.value()
        qspinbox_min = qspinbox.minimum()
        qspinbox_max = qspinbox.maximum()
        if isinstance(qspinbox_min, float):
            if np.isinf(qspinbox_max):
                qspinbox_max = sys.float_info.max
            value_range = np.linspace(qspinbox_min, qspinbox_max)
        else:
            # use + 1 to include maximum value
            value_range = range(qspinbox_min, qspinbox_max + 1)
            try:
                value_range_length = len(value_range)
            except OverflowError:
                # range too big for even trying to get how big it is.
                value_range_length = 100
                value_range = [
                    random.randrange(qspinbox_min, qspinbox_max)
                    for _ in range(value_range_length)
                ]
                value_range.append(qspinbox_max)
            if value_range_length > 100:
                # prevent iterating over a big range of values
                random.seed(0)
                value_range = random.sample(value_range, 100)
                value_range = np.insert(value_range, 0, qspinbox_min)
                value_range = np.append(value_range, qspinbox_max - 1)
        for value in value_range:
            qspinbox.setValue(value)
            # capture any output done to sys.stdout or sys.stderr.
            captured = capsys.readouterr()
            assert not captured.out
            if captured.err:
                # since an error was found check if it is associated with a known issue still open
                expected_errors = [
                    "MemoryError: Unable to allocate",  # See https://github.com/napari/napari/issues/5798
                    "ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.",  # See https://github.com/napari/napari/issues/5798
                    "ValueError: Maximum allowed dimension exceeded",  # See https://github.com/napari/napari/issues/5798
                    f"IndexError: index {value} is out of bounds for axis 0",  # See https://github.com/napari/napari/issues/4864
                    "RuntimeWarning: overflow encountered in scalar add",  # See https://github.com/napari/napari/issues/4864
                ]
                assert any(
                    expected_error in captured.err
                    for expected_error in expected_errors
                ), captured.err

        assert qspinbox.value() in [qspinbox_max, qspinbox_max - 1]
        qspinbox.setValue(qspinbox_initial_value)

    # check QAbstractSlider by changing value with `setValue` from minimum value to maximum
    for qslider in ctrl.findChildren(QAbstractSlider):
        if isinstance(qslider.minimum(), float):
            if getattr(qslider, "_valuesChanged", None):
                # create a list of tuples in the case the slider is ranged
                # from (minimum, minimum) to (maximum, maximum) +
                # from (minimum, maximum) to (minimum, minimum)
                # (minimum, minimum) and (maximum, maximum) values are excluded
                # to prevent the sequence not being monotonically increasing
                base_value_range = np.linspace(
                    qslider.minimum(), qslider.maximum()
                )
                num_values = base_value_range.size
                max_value = np.full(num_values, qslider.maximum())
                min_value = np.full(num_values, qslider.minimum())
                value_range_to_max = list(zip(base_value_range, max_value))
                value_range_to_min = list(
                    zip(min_value, np.flip(base_value_range))
                )
                value_range = value_range_to_max[:-1] + value_range_to_min[:-1]
            else:
                value_range = np.linspace(qslider.minimum(), qslider.maximum())
        else:
            if getattr(qslider, "_valuesChanged", None):
                # create a list of tuples in the case the slider is ranged
                # from (minimum, minimum) to (maximum, maximum) +
                # from (minimum, maximum) to (minimum, minimum)
                # base list created with + 1 to include maximum value
                # (minimum, minimum) and (maximum, maximum) values are excluded
                # to prevent the sequence not being monotonically increasing
                base_value_range = range(
                    qslider.minimum(), qslider.maximum() + 1
                )
                num_values = len(base_value_range)
                max_value = [qslider.maximum()] * num_values
                min_value = [qslider.minimum()] * num_values
                value_range_to_max = list(zip(base_value_range, max_value))
                base_value_range_copy = base_value_range.copy()
                base_value_range_copy.reverse()
                value_range_to_min = list(
                    zip(min_value, base_value_range_copy)
                )
                value_range = value_range_to_max[:-1] + value_range_to_min[:-1]
            else:
                # use + 1 to include maximum value
                value_range = range(qslider.minimum(), qslider.maximum() + 1)
        for value in value_range:
            qslider.setValue(value)
            # capture any output done to sys.stdout or sys.stderr.
            captured = capsys.readouterr()
            assert not captured.out
            assert not captured.err
        if getattr(qslider, "_valuesChanged", None):
            assert qslider.value()[0] == qslider.minimum()
        else:
            assert qslider.value() == qslider.maximum()

    # check QColorSwatchEdit by changing line edit text with a range of predefined values
    for qcolorswatchedit in ctrl.findChildren(QColorSwatchEdit):
        lineedit = qcolorswatchedit.line_edit
        colorswatch = qcolorswatchedit.color_swatch
        colors = [
            ("white", "white", np.array([1.0, 1.0, 1.0, 1.0])),
            ("black", "black", np.array([0.0, 0.0, 0.0, 1.0])),
            # check autocompletion `bla` -> `black`
            ("bla", "black", np.array([0.0, 0.0, 0.0, 1.0])),
            # check that setting an invalid color makes it fallback to the previous value
            ("invalid_value", "black", np.array([0.0, 0.0, 0.0, 1.0])),
        ]
        for color, expected_color, expected_array in colors:
            lineedit.clear()
            qtbot.keyClicks(lineedit, color)
            qtbot.keyClick(lineedit, Qt.Key_Enter)
            assert lineedit.text() == expected_color
            assert (colorswatch.color == expected_array).all()
            # capture any output done to sys.stdout or sys.stderr.
            captured = capsys.readouterr()
            assert not captured.out
            assert not captured.err

    # check QCheckBox by clicking with mouse click
    for qcheckbox in ctrl.findChildren(QCheckBox):
        if qcheckbox.isVisible():
            qcheckbox_checked = qcheckbox.isChecked()
            qtbot.mouseClick(qcheckbox, Qt.LeftButton)
            assert qcheckbox.isChecked() != qcheckbox_checked
            # capture any output done to sys.stdout or sys.stderr.
            captured = capsys.readouterr()
            assert not captured.out
            assert not captured.err

    # check QPushButton and QRadioButton by clicking with mouse click
    for button in ctrl.findChildren(QPushButton) + ctrl.findChildren(
        QRadioButton
    ):
        if button.isVisible():
            qtbot.mouseClick(button, Qt.LeftButton)
            # capture any output done to sys.stdout or sys.stderr.
            captured = capsys.readouterr()
            assert not captured.out
            assert not captured.err


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

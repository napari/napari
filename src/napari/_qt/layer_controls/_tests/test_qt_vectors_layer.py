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


def test_building_controls_leaves_the_layer_unchanged(qtbot):
    layer = Vectors(
        _VECTORS,
        features={'phase': np.array([0.5, 1.5])},
        edge_color='yellow',
    )

    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)

    assert layer.edge_color_mode == 'direct'
    np.testing.assert_allclose(layer.edge_color, [[1, 1, 0, 1], [1, 1, 0, 1]])


def test_mode_change_from_controls_reaches_other_listeners(qtbot):
    layer = Vectors(
        _VECTORS,
        features={'phase': np.array([0.5, 1.5])},
        edge_color='phase',
    )
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    heard = []
    layer.events.edge_color_mode.connect(lambda event: heard.append(event))

    control = qtctrl._edge_color_feature_control
    control.color_mode_combobox.setCurrentText('direct')

    assert layer.edge_color_mode == 'direct'
    assert len(heard) == 1


def test_colormap_controls_follow_the_mode(qtbot):
    """The colormap and its limits are shown, and editable, in colormap mode only."""
    layer = Vectors(_VECTORS)
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    control = qtctrl._edge_color_feature_control

    assert control.colormap_combobox.isHidden()
    assert control.contrast_limits_slider.isHidden()

    layer.features = {'phase': np.array([-np.pi, np.pi])}
    layer.edge_color = 'phase'
    layer.edge_color_mode = 'colormap'
    assert not control.colormap_combobox.isHidden()
    assert not control.contrast_limits_slider.isHidden()

    # widget -> layer
    index = control.colormap_combobox.findData('twilight')
    control.colormap_combobox.setCurrentIndex(index)
    control.contrast_limits_slider.setValue((-1.0, 1.0))
    assert layer.edge_colormap.name == 'twilight'
    np.testing.assert_allclose(layer.edge_contrast_limits, (-1.0, 1.0))

    # layer -> widget
    layer.edge_colormap = 'hsv'
    layer.edge_contrast_limits = (-2.0, 2.0)
    assert control.colormap_combobox.currentData() == 'hsv'
    np.testing.assert_allclose(
        control.contrast_limits_slider.value(), (-2.0, 2.0)
    )

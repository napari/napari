import numpy as np
import pytest

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
    """Constructing the controls must not assign a feature as edge_color.

    Populating the feature dropdown emits currentTextChanged for the first
    item; if the handler is already connected, building the controls assigns
    that feature as the layer's edge color, replacing an explicitly requested
    direct color with feature-mapped colors (the color assertion below is the
    regression detector; the mode is restored by change_edge_color_feature).
    """
    layer = Vectors(
        _VECTORS,
        features={'phase': np.array([0.5, 1.5])},
        edge_color='yellow',
    )
    assert layer.edge_color_mode == 'direct'

    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)

    assert layer.edge_color_mode == 'direct'
    np.testing.assert_allclose(layer.edge_color, [[1, 1, 0, 1], [1, 1, 0, 1]])


def test_mode_changes_from_the_controls_reach_other_listeners(qtbot):
    """A mode switch made in the combobox must emit edge_color_mode.

    change_edge_color_mode blocks its own resync callback while assigning the
    mode; blocking the whole emitter instead hides every combobox-driven mode
    switch from external listeners synchronizing state to the layer.
    """
    layer = Vectors(
        _VECTORS,
        features={'phase': np.array([0.5, 1.5])},
        edge_color='phase',
    )
    assert layer.edge_color_mode == 'colormap'
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    heard = []
    layer.events.edge_color_mode.connect(
        lambda event: heard.append(str(layer.edge_color_mode))
    )

    combobox = qtctrl._edge_color_feature_control.color_mode_combobox
    combobox.setCurrentText('direct')

    assert layer.edge_color_mode == 'direct'
    assert heard == ['direct']
    assert combobox.currentText() == 'direct'


def test_rejected_mode_change_rolls_back_without_notifying(qtbot):
    """A mode the layer refuses must not emit edge_color_mode to listeners.

    Selecting a mapped mode on a featureless layer raises; the layer never
    changed, so listeners must hear nothing, and the combobox must return to
    the mode the layer is actually in.
    """
    layer = Vectors(_VECTORS)
    qtctrl = QtVectorsControls(layer)
    qtbot.addWidget(qtctrl)
    control = qtctrl._edge_color_feature_control
    heard = []
    layer.events.edge_color_mode.connect(lambda event: heard.append(True))

    with pytest.raises(ValueError, match='valid Points'):
        control.change_edge_color_mode('colormap')

    assert layer.edge_color_mode == 'direct'
    assert control.color_mode_combobox.currentText() == 'direct'
    assert heard == []

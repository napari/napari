import numpy as np

from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari.layers import Vectors

_VECTORS = np.zeros((2, 2, 2))


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

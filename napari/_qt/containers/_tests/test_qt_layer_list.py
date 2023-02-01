import numpy as np
from qtpy.QtCore import Qt

from napari._qt.containers import QtLayerList
from napari.components import LayerList
from napari.layers import Image


def test_set_layer_visible_changes_checked_state(qtbot):
    image = Image(np.zeros((4, 3)))
    layers = LayerList([image])
    qt_layers_view = QtLayerList(layers)
    qtbot.addWidget(qt_layers_view)
    assert image.visible
    assert (
        check_state_at_layer_index(qt_layers_view, 0) == Qt.CheckState.Checked
    )

    image.visible = False

    assert (
        check_state_at_layer_index(qt_layers_view, 0)
        == Qt.CheckState.Unchecked
    )


def check_state_at_layer_index(
    qt_layers_view: QtLayerList, index: int
) -> Qt.CheckState:
    qt_layers_model = qt_layers_view.model()
    qt_layer_index = qt_layers_model.index(
        index, 0, qt_layers_view.rootIndex()
    )
    return qt_layers_model.data(qt_layer_index, Qt.ItemDataRole.CheckStateRole)

import numpy as np
from qtpy.QtCore import Qt

from napari._qt.containers import QtLayerList
from napari.components import LayerList
from napari.layers import Image


def test_set_layer_visible_changes_checked_state(qtbot):
    image = Image(np.zeros((4, 3)))
    layers = LayerList([image])
    view = QtLayerList(layers)
    qtbot.addWidget(view)
    assert image.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Checked

    image.visible = False

    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Unchecked


def check_state_at_layer_index(
    view: QtLayerList, layer_index: int
) -> Qt.CheckState:
    model = view.model()
    model_index = model.index(layer_index, 0, view.rootIndex())
    return model.data(model_index, Qt.ItemDataRole.CheckStateRole)

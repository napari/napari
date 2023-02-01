from typing import Tuple

import numpy as np
from qtpy.QtCore import QModelIndex, Qt

from napari._qt.containers import QtLayerList
from napari.components import LayerList
from napari.layers import Image


def test_set_layer_invisible_makes_item_unchecked(qtbot):
    view, image = make_qt_layer_list_with_layer(qtbot)
    assert image.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Checked

    image.visible = False

    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Unchecked


def test_set_item_unchecked_makes_layer_invisible(qtbot):
    view, image = make_qt_layer_list_with_layer(qtbot)
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Checked
    assert image.visible

    view.model().setData(
        layer_to_model_index(view, 0),
        Qt.CheckState.Unchecked,
        Qt.ItemDataRole.CheckStateRole,
    )

    assert not image.visible


def make_qt_layer_list_with_layer(qtbot) -> Tuple[QtLayerList, Image]:
    image = Image(np.zeros((4, 3)))
    layers = LayerList([image])
    view = QtLayerList(layers)
    qtbot.addWidget(view)
    return view, image


def layer_to_model_index(view: QtLayerList, layer_index: int) -> QModelIndex:
    return view.model().index(layer_index, 0, view.rootIndex())


def check_state_at_layer_index(
    view: QtLayerList, layer_index: int
) -> Qt.CheckState:
    model_index = layer_to_model_index(view, layer_index)
    return view.model().data(model_index, Qt.ItemDataRole.CheckStateRole)

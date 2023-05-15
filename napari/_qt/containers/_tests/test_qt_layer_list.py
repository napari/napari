from typing import List, Tuple

import numpy as np
from qtpy.QtCore import QModelIndex, QPoint, Qt

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


def test_drag_and_drop_layers(qtbot):
    view, images = make_qt_layer_list_with_layers(qtbot)
    name = view.model().data(
        layer_to_model_index(view, 0), Qt.ItemDataRole.DisplayRole
    )
    assert name == "image2"

    # drag event
    qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    qtbot.mouseMoved(view, QPoint(100, 100))
    qtbot.mouseRelease(view, Qt.MouseButton.LeftButton)

    name = view.model().data(
        layer_to_model_index(view, 0), Qt.ItemDataRole.DisplayRole
    )
    assert name == "image1"


def make_qt_layer_list_with_layer(qtbot) -> Tuple[QtLayerList, Image]:
    image = Image(np.zeros((4, 3)))
    layers = LayerList([image])
    view = QtLayerList(layers)
    qtbot.addWidget(view)
    return view, image


def make_qt_layer_list_with_layers(qtbot) -> Tuple[QtLayerList, List[Image]]:
    image1 = Image(np.zeros((4, 3)), name="image1")
    image2 = Image(np.zeros((4, 3)), name="image2")
    layers = LayerList([image1, image2])
    view = QtLayerList(layers)
    qtbot.addWidget(view)
    return view, [image1, image2]


def layer_to_model_index(view: QtLayerList, layer_index: int) -> QModelIndex:
    return view.model().index(layer_index, 0, view.rootIndex())


def check_state_at_layer_index(
    view: QtLayerList, layer_index: int
) -> Qt.CheckState:
    model_index = layer_to_model_index(view, layer_index)
    value = view.model().data(model_index, Qt.ItemDataRole.CheckStateRole)
    # The data method returns integer value of the enum in some cases, so
    # ensure it has the enum type for more explicit assertions.
    return Qt.CheckState(value)

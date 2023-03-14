from typing import Tuple

import numpy as np
from qtpy.QtCore import QModelIndex, Qt

from napari._qt.containers import QtLayerList
from napari._qt.containers._layer_delegate import LayerDelegate
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


def test_alt_click_to_show_single_layer(qtbot):
    image1 = Image(np.zeros((4, 3)))
    image2 = Image(np.zeros((4, 3)))
    image3 = Image(np.zeros((4, 3)))

    layers = LayerList([image1, image2, image3])
    # this will make the list have image2 on top of image1
    view = QtLayerList(layers)
    qtbot.addWidget(view)

    delegate = LayerDelegate()

    # hide the middle-layer, image2 and ensure it's unchecked
    image2.visible = False
    assert check_state_at_layer_index(view, 1) == Qt.CheckState.Unchecked
    # ensure the other layers are visible, checked
    assert image3.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Checked
    assert image1.visible
    assert check_state_at_layer_index(view, 2) == Qt.CheckState.Checked

    # mock an alt-click on top-most layer, image3
    index = layer_to_model_index(view, 0)
    delegate._show_on_alt_click_hide_others(view.model(), index)

    # only image3 should be shown, while image1, image2 be hidden
    assert image3.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Checked
    assert not image1.visible
    assert check_state_at_layer_index(view, 2) == Qt.CheckState.Unchecked
    assert not image2.visible
    assert check_state_at_layer_index(view, 1) == Qt.CheckState.Unchecked

    # mock second alt-click, which should restore initial state
    delegate._show_on_alt_click_hide_others(view.model(), index)

    assert image3.visible
    assert not image2.visible
    assert image1.visible


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
    value = view.model().data(model_index, Qt.ItemDataRole.CheckStateRole)
    # The data method returns integer value of the enum in some cases, so
    # ensure it has the enum type for more explicit assertions.
    return Qt.CheckState(value)

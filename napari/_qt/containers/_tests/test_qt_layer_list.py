from typing import List, Tuple

import numpy as np
import pyautogui
from qtpy.QtCore import QModelIndex, QPoint, Qt, QVariantAnimation

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
    view.show()

    # check initial element is the one expected (last element in the layerlist)
    name = view.model().data(
        layer_to_model_index(view, 0), Qt.ItemDataRole.DisplayRole
    )
    assert name == images[-1].name

    # drag event simulation
    base_pos = view.mapToGlobal(view.rect().topLeft())
    start_pos = base_pos + QPoint(10, 10)
    end_pos = base_pos + QPoint(100, 100)

    def on_animation_value_changed(value):
        pyautogui.moveTo(value.x(), value.y())
        if value == end_pos:
            pyautogui.mouseUp(button="left")

    animation = QVariantAnimation(
        startValue=start_pos, endValue=end_pos, duration=5000
    )
    animation.valueChanged.connect(on_animation_value_changed)

    pyautogui.moveTo(start_pos.x(), start_pos.y())
    pyautogui.mouseDown(button="left")
    with qtbot.waitSignal(animation.finished, timeout=10000):
        animation.start()

    name = view.model().data(
        layer_to_model_index(view, 0), Qt.ItemDataRole.DisplayRole
    )
    assert name == images[0].name


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

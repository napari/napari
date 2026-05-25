import threading

import numpy as np
from qtpy.QtCore import QPoint, Qt

from napari._qt.containers import QtLayerList
from napari._qt.containers._tests.test_qt_layer_list import (
    layer_to_model_index,
)
from napari._tests.utils import skip_local_focus
from napari.components import LayerList
from napari.layers import Image


def drag_and_drop(start_x, start_y, end_x, end_y):
    # simulate a drag and drop action with pyautogui
    import pyautogui

    pyautogui.moveTo(start_x, start_y, duration=0.2)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=0.2)
    pyautogui.mouseUp()


def make_qt_layer_list_with_layers(qtbot) -> tuple[QtLayerList, list[Image]]:
    image1 = Image(np.zeros((4, 3)), name='image1')
    image2 = Image(np.zeros((4, 3)), name='image2')
    layers = LayerList([image1, image2])
    view = QtLayerList(layers)
    qtbot.addWidget(view)
    return view, [image1, image2]


@skip_local_focus
def test_drag_and_drop_layers(qtbot):
    """
    Test drag and drop actions with pyautogui to change layer list order.

    Notes:
        * For this test to pass locally on macOS, you need to give the Terminal/iTerm
          application accessibility permissions:
              `System Settings > Privacy & Security > Accessibility`

        See https://github.com/asweigart/pyautogui/issues/247 and
        https://github.com/asweigart/pyautogui/issues/247#issuecomment-437668855.
    """
    view, images = make_qt_layer_list_with_layers(qtbot)
    with qtbot.waitExposed(view):
        view.show()

    # check initial element is the one expected (last element in the layerlist)
    name = view.model().data(
        layer_to_model_index(view, 0), Qt.ItemDataRole.DisplayRole
    )
    assert name == images[-1].name

    # drag and drop event simulation
    base_pos = view.mapToGlobal(view.rect().topLeft())
    start_pos = base_pos + QPoint(50, 10)
    start_x = start_pos.x()
    start_y = start_pos.y()
    end_pos = base_pos + QPoint(100, 100)
    end_x = end_pos.x()
    end_y = end_pos.y()

    drag_drop = threading.Thread(
        target=drag_and_drop, args=(start_x, start_y, end_x, end_y)
    )
    drag_drop.start()

    def check_drag_and_drop():
        # check layerlist first element corresponds with first layer in the GUI
        name = view.model().data(
            layer_to_model_index(view, 0), Qt.ItemDataRole.DisplayRole
        )
        return name == images[0].name

    qtbot.waitUntil(check_drag_and_drop)

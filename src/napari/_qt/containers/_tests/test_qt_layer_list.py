import threading

import numpy as np
from qtpy.QtCore import QModelIndex, QPoint, Qt
from qtpy.QtWidgets import QLineEdit, QStyleOptionViewItem

from napari._qt.containers import QtLayerList
from napari._qt.containers._layer_delegate import LayerDelegate
from napari._tests.utils import skip_local_focus
from napari.components import LayerList
from napari.layers import Image, Shapes


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
    (
        image1,
        image2,
        image3,
        layers,
        view,
        delegate,
    ) = make_qt_layer_list_with_delegate(qtbot)

    # hide the middle-layer, image2 and ensure it's unchecked
    image2.visible = False
    assert check_state_at_layer_index(view, 1) == Qt.CheckState.Unchecked

    # ensure the other layers are visible & checked
    assert image3.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Checked
    assert image1.visible
    assert check_state_at_layer_index(view, 2) == Qt.CheckState.Checked

    # alt-click state should be None
    assert delegate._alt_click_layer() is None

    # mock an alt-click on bottom-most layer, image1
    index = layer_to_model_index(view, 2)
    delegate._show_on_alt_click_hide_others(view.model(), index)

    # alt-click state should be set to image1
    assert delegate._alt_click_layer() == image1

    # only image1 should be shown, while image3, image2 be hidden
    assert not image3.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Unchecked
    assert not image2.visible
    assert check_state_at_layer_index(view, 1) == Qt.CheckState.Unchecked
    assert image1.visible
    assert check_state_at_layer_index(view, 2) == Qt.CheckState.Checked


def test_second_alt_click_to_show_different_layer(qtbot):
    (
        image1,
        image2,
        image3,
        layers,
        view,
        delegate,
    ) = make_qt_layer_list_with_delegate(qtbot)

    # mock an alt-click on bottom-most layer, image1
    index = layer_to_model_index(view, 2)
    delegate._show_on_alt_click_hide_others(view.model(), index)

    # alt-click state should be set to image1
    assert delegate._alt_click_layer() == image1

    # image2 should be hidden
    assert not image2.visible
    assert check_state_at_layer_index(view, 1) == Qt.CheckState.Unchecked

    # mock an alt-click on middle layer, image2
    index2 = layer_to_model_index(view, 1)
    delegate._show_on_alt_click_hide_others(view.model(), index2)

    # alt-click state should be set to image2
    assert delegate._alt_click_layer() == image2

    # only image2 should be shown, while image3, image1 be hidden
    assert not image3.visible
    assert check_state_at_layer_index(view, 0) == Qt.CheckState.Unchecked
    assert not image1.visible
    assert check_state_at_layer_index(view, 2) == Qt.CheckState.Unchecked
    assert image2.visible
    assert check_state_at_layer_index(view, 1) == Qt.CheckState.Checked


def test_second_alt_click_to_restore_layer_state(qtbot):
    (
        image1,
        image2,
        image3,
        layers,
        view,
        delegate,
    ) = make_qt_layer_list_with_delegate(qtbot)

    # mock an alt-click on bottom-most layer, image1
    index = layer_to_model_index(view, 2)
    delegate._show_on_alt_click_hide_others(view.model(), index)

    # add a layer (will be at position 0)
    image4 = Image(np.zeros((4, 3)))
    layers.append(image4)
    assert image4.visible

    # remove a layer (image3, which has been pushed down to position 1
    layers.pop(1)

    # mock second alt-click on image1, which should restore initial state
    delegate._show_on_alt_click_hide_others(view.model(), index)

    # image4 should remain visible (not part of initial state)
    # image2 should be not visible--that was the initial state
    assert image4.visible
    assert image1.visible
    assert not image2.visible

    # alt-click state should be cleared
    assert delegate._alt_click_layer() is None


def test_contextual_menu_updates_selection_ctx_keys(monkeypatch, qtbot):
    shapes_layer = Shapes()
    layer_list = LayerList()
    layer_list._create_contexts()
    layer_list.append(shapes_layer)
    view = QtLayerList(layer_list)
    qtbot.addWidget(view)
    delegate = view.itemDelegate()
    assert not layer_list[0].data

    layer_list.selection.add(shapes_layer)
    index = layer_to_model_index(view, 0)
    assert layer_list._selection_ctx_keys.num_selected_shapes_layers == 1
    assert layer_list._selection_ctx_keys.selected_empty_shapes_layer

    monkeypatch.setattr(
        'app_model.backends.qt.QModelMenu.exec_', lambda self, x: x
    )

    delegate.show_context_menu(
        index, view.model(), QPoint(10, 10), parent=view
    )
    assert not delegate._context_menu.findAction(
        'napari.layer.convert_to_labels'
    ).isEnabled()

    layer_list[0].add(np.array(([0, 0], [0, 10], [10, 10], [10, 0])))
    assert layer_list[0].data
    delegate.show_context_menu(
        index, view.model(), QPoint(10, 10), parent=view
    )
    assert delegate._context_menu.findAction(
        'napari.layer.convert_to_labels'
    ).isEnabled()


def make_qt_layer_list_with_delegate(qtbot):
    image1 = Image(np.zeros((4, 3)))
    image2 = Image(np.zeros((4, 3)))
    image3 = Image(np.zeros((4, 3)))

    layers = LayerList([image1, image2, image3])
    # this will make the list have image2 on top of image1
    view = QtLayerList(layers)
    qtbot.addWidget(view)

    delegate = LayerDelegate()
    return image1, image2, image3, layers, view, delegate


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


def drag_and_drop(start_x, start_y, end_x, end_y):
    # simulate a drag and drop action with pyautogui
    import pyautogui

    pyautogui.moveTo(start_x, start_y, duration=0.2)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=0.2)
    pyautogui.mouseUp()


def make_qt_layer_list_with_layer(qtbot) -> tuple[QtLayerList, Image]:
    image = Image(np.zeros((4, 3)))
    layers = LayerList([image])
    view = QtLayerList(layers)
    qtbot.addWidget(view)
    return view, image


def make_qt_layer_list_with_layers(qtbot) -> tuple[QtLayerList, list[Image]]:
    image1 = Image(np.zeros((4, 3)), name='image1')
    image2 = Image(np.zeros((4, 3)), name='image2')
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


def test_createEditor(qtbot):
    view, image = make_qt_layer_list_with_layer(qtbot)
    model_index = layer_to_model_index(view, 0)
    delegate = view.itemDelegate()
    editor = delegate.createEditor(view, QStyleOptionViewItem(), model_index)
    assert isinstance(editor, QLineEdit)
    delegate.setEditorData(editor, model_index)
    assert editor.text() == image.name

import numpy as np
from qtpy.QtCore import QModelIndex, QPoint, Qt
from qtpy.QtWidgets import QLineEdit, QStyleOptionViewItem

from napari._qt.containers import QtLayerList
from napari._qt.containers._layer_delegate import LayerDelegate
from napari._qt.containers.qt_layer_model import LockedRole
from napari.components import LayerList
from napari.layers import Image, Labels, Shapes


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
        _layers,
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
        _layers,
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
        _image3,
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
    from napari._app_model import get_app_model

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

    app = get_app_model()
    with app.injection_store.register(providers={LayerList: layer_list}):
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


def _lock_blocks_action(qtbot, monkeypatch, layers_factory, action_id):
    from napari._app_model import get_app_model

    layer_list, lock_target = layers_factory()
    layer_list._create_contexts()
    view = QtLayerList(layer_list)
    qtbot.addWidget(view)
    delegate = view.itemDelegate()
    for lay in layer_list:
        layer_list.selection.add(lay)
    index = layer_to_model_index(view, 0)

    monkeypatch.setattr(
        'app_model.backends.qt.QModelMenu.exec_', lambda self, x: x
    )
    app = get_app_model()

    def menu_action_enabled():
        with app.injection_store.register(providers={LayerList: layer_list}):
            delegate.show_context_menu(
                index, view.model(), QPoint(10, 10), parent=view
            )
            return delegate._context_menu.findAction(action_id).isEnabled()

    assert menu_action_enabled()
    lock_target.locked = True
    assert not menu_action_enabled()
    lock_target.locked = False
    assert menu_action_enabled()


def test_locked_blocks_convert_to_labels(qtbot, monkeypatch):
    def factory():
        layer_list = LayerList()
        image = Image(np.zeros((4, 3)))
        layer_list.append(image)
        return layer_list, image

    _lock_blocks_action(
        qtbot, monkeypatch, factory, 'napari.layer.convert_to_labels'
    )


def test_locked_blocks_convert_to_image(qtbot, monkeypatch):
    def factory():
        layer_list = LayerList()
        labels = Labels(np.zeros((4, 3), dtype=int))
        layer_list.append(labels)
        return layer_list, labels

    _lock_blocks_action(
        qtbot, monkeypatch, factory, 'napari.layer.convert_to_image'
    )


def test_locked_blocks_split_stack(qtbot, monkeypatch):
    def factory():
        layer_list = LayerList()
        image = Image(np.zeros((4, 3, 3)))
        layer_list.append(image)
        return layer_list, image

    _lock_blocks_action(
        qtbot, monkeypatch, factory, 'napari.layer.split_stack'
    )


def test_locked_blocks_split_rgb(qtbot, monkeypatch):
    def factory():
        layer_list = LayerList()
        image = Image(np.zeros((4, 3, 3)), rgb=True)
        layer_list.append(image)
        return layer_list, image

    _lock_blocks_action(qtbot, monkeypatch, factory, 'napari.layer.split_rgb')


def test_locked_blocks_merge_stack(qtbot, monkeypatch):
    def factory():
        layer_list = LayerList()
        image_a = Image(np.zeros((4, 3)))
        image_b = Image(np.zeros((4, 3)))
        layer_list.append(image_a)
        layer_list.append(image_b)
        return layer_list, image_a

    _lock_blocks_action(
        qtbot, monkeypatch, factory, 'napari.layer.merge_stack'
    )


def test_locked_blocks_merge_rgb(qtbot, monkeypatch):
    def factory():
        layer_list = LayerList()
        image_a = Image(np.zeros((4, 3)))
        image_b = Image(np.zeros((4, 3)))
        image_c = Image(np.zeros((4, 3)))
        layer_list.append(image_a)
        layer_list.append(image_b)
        layer_list.append(image_c)
        return layer_list, image_a

    _lock_blocks_action(qtbot, monkeypatch, factory, 'napari.layer.merge_rgb')


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


def make_qt_layer_list_with_layer(qtbot) -> tuple[QtLayerList, Image]:
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


def test_createEditor(qtbot):
    view, image = make_qt_layer_list_with_layer(qtbot)
    model_index = layer_to_model_index(view, 0)
    delegate = view.itemDelegate()
    editor = delegate.createEditor(view, QStyleOptionViewItem(), model_index)
    assert isinstance(editor, QLineEdit)
    delegate.setEditorData(editor, model_index)
    assert editor.text() == image.name


def test_lock_role_data(qtbot):
    """LockedRole should return the layer's locked property."""
    view, image = make_qt_layer_list_with_layer(qtbot)
    model_index = layer_to_model_index(view, 0)
    assert not view.model().data(model_index, LockedRole)
    image.locked = True
    assert view.model().data(model_index, LockedRole)


def test_lock_role_set_data(qtbot):
    """setData with LockedRole should change layer.locked."""
    view, image = make_qt_layer_list_with_layer(qtbot)
    model_index = layer_to_model_index(view, 0)
    view.model().setData(model_index, True, LockedRole)
    assert image.locked


def test_process_event_locked(qtbot):
    """locked event should trigger dataChanged signal."""
    view, image = make_qt_layer_list_with_layer(qtbot)
    changed_signals = []
    view.model().dataChanged.connect(lambda *a: changed_signals.append(a))
    image.locked = True
    assert len(changed_signals) >= 1


def test_paint_lock_icon_locked(qtbot):
    """Painting locked layer should not raise errors."""
    view, image = make_qt_layer_list_with_layer(qtbot)
    image.locked = True
    view.viewport().update()
    qtbot.wait(100)


def test_paint_lock_icon_unlocked(qtbot):
    """Painting unlocked layer should not raise errors."""
    view, _image = make_qt_layer_list_with_layer(qtbot)
    view.viewport().update()
    qtbot.wait(100)

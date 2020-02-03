import numpy as np
from napari.layers import Shapes
from napari.layers.shapes import keybindings


def test_lock_aspect_ratio():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    layer._is_moving = True
    # need to go through the generator
    _ = list(keybindings.hold_to_lock_aspect_ratio(layer))


def test_lock_aspect_ratio_selected_box():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    # select a shape
    layer._selected_box = layer.interaction_box(0)
    layer._is_moving = True
    # need to go through the generator
    _ = list(keybindings.hold_to_lock_aspect_ratio(layer))


def test_lock_aspect_ratio_selected_box_zeros():
    # Test a single four corner rectangle that has zero size
    layer = Shapes(20 * np.zeros((1, 4, 2)))
    # select a shape
    layer._selected_box = layer.interaction_box(0)
    layer._is_moving = True
    # need to go through the generator
    _ = list(keybindings.hold_to_lock_aspect_ratio(layer))


def test_hold_to_pan_zoom():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    layer.mode = 'direct'
    # need to go through the generator
    _ = list(keybindings.hold_to_pan_zoom(layer))


def test_activate_modes():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    # need to go through the generator
    keybindings.activate_add_rectangle_mode(layer)
    assert layer.mode == 'add_rectangle'
    keybindings.activate_add_ellipse_mode(layer)
    assert layer.mode == 'add_ellipse'
    keybindings.activate_add_line_mode(layer)
    assert layer.mode == 'add_line'
    keybindings.activate_add_path_mode(layer)
    assert layer.mode == 'add_path'
    keybindings.activate_add_polygon_mode(layer)
    assert layer.mode == 'add_polygon'
    keybindings.activate_direct_mode(layer)
    assert layer.mode == 'direct'
    keybindings.activate_select_mode(layer)
    assert layer.mode == 'select'
    keybindings.activate_pan_zoom_mode(layer)
    assert layer.mode == 'pan_zoom'
    keybindings.activate_vertex_insert_mode(layer)
    assert layer.mode == 'vertex_insert'
    keybindings.activate_vertex_remove_mode(layer)
    assert layer.mode == 'vertex_remove'


def test_copy_paste():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    layer.mode = 'direct'

    assert len(layer.data) == 3
    assert layer._clipboard == {}
    layer.selected_data = [0, 1]

    keybindings.copy(layer)
    assert len(layer.data) == 3
    assert len(layer._clipboard) == 2

    keybindings.paste(layer)
    assert len(layer.data) == 5
    assert len(layer._clipboard) == 2


def test_select_all():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    layer.mode = 'direct'

    assert len(layer.data) == 3
    assert len(layer.selected_data) == 0

    keybindings.select_all(layer)
    assert len(layer.selected_data) == 3


def test_delete():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    layer.mode = 'direct'

    assert len(layer.data) == 3
    layer.selected_data = [0, 1]

    keybindings.delete_selected(layer)
    assert len(layer.data) == 1


def test_finish():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    keybindings.finish_drawing(layer)

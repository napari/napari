import numpy as np

from napari.layers import Shapes
from napari.layers.shapes import _shapes_key_bindings as key_bindings


def test_lock_aspect_ratio():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    layer._moving_coordinates = (0, 0, 0)
    layer._is_moving = True
    # need to go through the generator
    _ = list(key_bindings.hold_to_lock_aspect_ratio(layer))


def test_lock_aspect_ratio_selected_box():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    # select a shape
    layer._selected_box = layer.interaction_box(0)
    layer._moving_coordinates = (0, 0, 0)
    layer._is_moving = True
    # need to go through the generator
    _ = list(key_bindings.hold_to_lock_aspect_ratio(layer))


def test_lock_aspect_ratio_selected_box_zeros():
    # Test a single four corner rectangle that has zero size
    layer = Shapes(20 * np.zeros((1, 4, 2)))
    # select a shape
    layer._selected_box = layer.interaction_box(0)
    layer._moving_coordinates = (0, 0, 0)
    layer._is_moving = True
    # need to go through the generator
    _ = list(key_bindings.hold_to_lock_aspect_ratio(layer))


def test_activate_modes():
    # Test a single four corner rectangle
    layer = Shapes(20 * np.random.random((1, 4, 2)))
    # need to go through the generator
    key_bindings.activate_add_rectangle_mode(layer)
    assert layer.mode == 'add_rectangle'
    key_bindings.activate_add_ellipse_mode(layer)
    assert layer.mode == 'add_ellipse'
    key_bindings.activate_add_line_mode(layer)
    assert layer.mode == 'add_line'
    key_bindings.activate_add_path_mode(layer)
    assert layer.mode == 'add_path'
    key_bindings.activate_add_polygon_mode(layer)
    assert layer.mode == 'add_polygon'
    key_bindings.activate_direct_mode(layer)
    assert layer.mode == 'direct'
    key_bindings.activate_select_mode(layer)
    assert layer.mode == 'select'
    key_bindings.activate_shape_pan_zoom_mode(layer)
    assert layer.mode == 'pan_zoom'
    key_bindings.activate_vertex_insert_mode(layer)
    assert layer.mode == 'vertex_insert'
    key_bindings.activate_vertex_remove_mode(layer)
    assert layer.mode == 'vertex_remove'


def test_copy_paste():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    layer.mode = 'direct'

    assert len(layer.data) == 3
    assert layer._clipboard == {}
    layer.selected_data = {0, 1}

    key_bindings.copy_selected_shapes(layer)
    assert len(layer.data) == 3
    assert len(layer._clipboard) > 0

    key_bindings.paste_shape(layer)
    assert len(layer.data) == 5
    assert len(layer._clipboard) > 0


def test_select_all():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    layer.mode = 'direct'

    assert len(layer.data) == 3
    assert len(layer.selected_data) == 0

    key_bindings.select_all_shapes(layer)
    assert len(layer.selected_data) == 3


def test_delete():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    layer.mode = 'direct'

    assert len(layer.data) == 3
    layer.selected_data = {0, 1}

    key_bindings.delete_selected_shapes(layer)
    assert len(layer.data) == 1


def test_finish():
    # Test on three four corner rectangle
    layer = Shapes(20 * np.random.random((3, 4, 2)))
    key_bindings.finish_drawing_shape(layer)

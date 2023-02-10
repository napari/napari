from napari.layers.points import Points
from napari.layers.points import _points_key_bindings as key_bindings


def test_modes(layer):
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)

    key_bindings.activate_points_add_mode(layer)
    assert layer.mode == 'add'
    key_bindings.activate_points_select_mode(layer)
    assert layer.mode == 'select'
    key_bindings.activate_points_pan_zoom_mode(layer)
    assert layer.mode == 'pan_zoom'


def test_copy_paste(layer):

    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'

    assert len(layer.data) == 4
    assert layer._clipboard == {}
    layer.selected_data = {0, 1}

    key_bindings.copy(layer)
    assert len(layer.data) == 4
    assert len(layer._clipboard) > 0

    key_bindings.paste(layer)
    assert len(layer.data) == 6
    assert len(layer._clipboard) > 0


def test_select_all_in_slice(layer):

    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'
    layer._set_view_slice()

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 4

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 0


def test_select_all_in_slice_3d_data(layer):

    data = [[0, 1, 3], [0, 8, 4], [0, 10, 10], [1, 15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'
    layer._set_view_slice()

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 3

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 0


def test_select_all_data(layer):

    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'
    layer._set_view_slice()

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 4

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 0


def test_select_all_data_3d_data(layer):

    data = [[0, 1, 3], [0, 8, 4], [0, 10, 10], [1, 15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'
    layer._set_view_slice()

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 4

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 0


def test_select_all_mixed(layer):
    data = [[0, 1, 3], [0, 8, 4], [0, 10, 10], [1, 15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'
    layer._set_view_slice()

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 4

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 1

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 4

    key_bindings.select_all_in_slice(layer)
    assert len(layer.selected_data) == 1

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 4

    key_bindings.select_all_data(layer)
    assert len(layer.selected_data) == 0


def test_delete_selected_points(layer):
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)
    layer.mode = 'select'

    assert len(layer.data) == 4
    layer.selected_data = {0, 1}

    key_bindings.delete_selected_points(layer)
    assert len(layer.data) == 2

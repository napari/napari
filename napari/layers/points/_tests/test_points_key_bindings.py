from napari.layers import Points


def test_select_all():
    """Test select all key binding."""
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    layer.mode = 'select'
    layer.class_keymap['A'](layer)
    assert len(layer.selected_data) == 4

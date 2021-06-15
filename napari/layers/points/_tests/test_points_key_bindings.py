from napari.layers import Points


def test_select_all(make_napari_viewer):
    """Test select all key binding."""

    # make viewer to bind shortcuts
    viewer = make_napari_viewer()  # noqa: F841

    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)

    assert len(layer.data) == 4
    assert len(layer.selected_data) == 0

    layer.mode = 'select'
    layer.class_keymap['A'](layer)
    assert len(layer.selected_data) == 4

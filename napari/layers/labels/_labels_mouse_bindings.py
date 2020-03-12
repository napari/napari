from ._labels_utils import interpolate_coordinates


def paint(layer, event):
    """Paint with the currently selected label."""
    # on press
    layer._save_history()
    layer._block_saving = True
    layer.paint(layer.coordinates, layer.selected_label)
    last_cursor_coord = layer.coordinates
    yield

    # on move
    while event.type == 'mouse_move':
        interp_coord = interpolate_coordinates(
            last_cursor_coord, layer.coordinates, layer.brush_size
        )
        for c in interp_coord:
            layer.paint(c, layer.selected_label, refresh=False)
        layer.refresh()
        last_cursor_coord = layer.coordinates
        yield

    # on release
    layer._block_saving = False


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = layer._value or 0


def fill(layer, event):
    """Fill in an area with the currently selected label."""
    # on press
    layer.fill(layer.coordinates, layer._value, layer.selected_label)

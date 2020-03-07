from copy import copy

from ._labels_utils import interpolate_coordinates


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = layer._value or 0
    yield

    # on move
    while event.type == 'mouse_move':
        yield

    # on release
    layer._last_cursor_coord = None
    layer._block_saving = False


def paint(layer, event):
    """Paint with the currently selected label."""
    # on press
    layer._save_history()
    layer._block_saving = True
    layer.paint(layer.coordinates, layer.selected_label)
    layer._last_cursor_coord = copy(layer.coordinates)
    yield

    # on move
    while event.type == 'mouse_move':
        if layer._last_cursor_coord is None:
            interp_coord = [layer.coordinates]
        else:
            interp_coord = interpolate_coordinates(
                layer._last_cursor_coord, layer.coordinates, layer.brush_size
            )
        for c in interp_coord:
            layer.paint(c, layer.selected_label, refresh=False)
        layer.refresh()
        layer._last_cursor_coord = copy(layer.coordinates)
        yield

    # on release
    layer._last_cursor_coord = None
    layer._block_saving = False


def fill(layer, event):
    """Fill in an area with the currently selected label."""
    # on press
    layer.fill(layer.coordinates, layer._value, layer.selected_label)
    yield

    # on move
    while event.type == 'mouse_move':
        yield

    # on release
    layer._last_cursor_coord = None
    layer._block_saving = False

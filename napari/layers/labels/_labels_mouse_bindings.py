from ._labels_utils import interpolate_coordinates
from ._labels_constants import Mode


def paint(layer, event):
    """Paint with the proper label.

    Use current selected label when in paint mode,
    background label when in erase mode.
    """
    if layer._mode == Mode.ERASE or event.button == 2:
        label = layer._background_label
    else:
        label = layer.selected_label
    # on press
    layer._save_history()
    layer._block_saving = True
    layer.paint(layer.coordinates, label)
    last_cursor_coord = layer.coordinates
    yield

    # on move
    while event.type == 'mouse_move':
        interp_coord = interpolate_coordinates(
            last_cursor_coord, layer.coordinates, layer.brush_size
        )
        for c in interp_coord:
            layer.paint(c, label, refresh=False)
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
    if layer._mode == Mode.ERASE or event.button == 2:
        label = layer._background_label
    else:
        label = layer.selected_label
    # on press
    layer.fill(layer.coordinates, layer._value, label)

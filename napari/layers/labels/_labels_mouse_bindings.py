from ._labels_utils import interpolate_coordinates
from ._labels_constants import Mode


def draw(layer, event):
    """Paint with the currently selected label."""
    # on press
    layer._save_history()
    layer._block_saving = True
    if layer._mode == Mode.PAINT:
        layer.paint(layer.coordinates, layer.selected_label)
    elif layer._mode == Mode.FILL:
        layer.fill(layer.coordinates, layer._value, layer.selected_label)
    last_cursor_coord = layer.coordinates
    yield

    # on move
    while event.type == 'mouse_move':
        interp_coord = interpolate_coordinates(
            last_cursor_coord, layer.coordinates, layer.brush_size
        )
        for c in interp_coord:
            if layer._mode == Mode.PAINT:
                layer.paint(c, layer.selected_label, refresh=False)
            elif (
                layer._mode == Mode.FILL
                and layer._value != layer.selected_label
            ):
                layer.fill(
                    layer.coordinates, layer._value, layer.selected_label
                )
        layer.refresh()
        last_cursor_coord = layer.coordinates
        yield

    # on release
    layer._block_saving = False


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = layer._value or 0

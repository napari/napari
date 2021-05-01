from ._labels_constants import Mode
from ._labels_utils import interpolate_coordinates


def draw(layer, event):
    """Draw with the currently selected label to a coordinate.

    This method have different behavior when draw is called
    with different labeling layer mode.

    In PAINT mode the cursor functions like a paint brush changing any
    pixels it brushes over to the current label. If the background label
    `0` is selected than any pixels will be changed to background and this
    tool functions like an eraser. The size and shape of the cursor can be
    adjusted in the properties widget.

    In FILL mode the cursor functions like a fill bucket replacing pixels
    of the label clicked on with the current label. It can either replace
    all pixels of that label or just those that are contiguous with the
    clicked on pixel. If the background label `0` is selected than any
    pixels will be changed to background and this tool functions like an
    eraser
    """
    coordinates = layer.world_to_data(event.position)
    # on press
    if layer._mode == Mode.ERASE:
        new_label = layer._background_label
    else:
        new_label = layer.selected_label

    if layer._mode in [Mode.PAINT, Mode.ERASE]:
        layer.paint(coordinates, new_label)
    elif layer._mode == Mode.FILL:
        layer.fill(coordinates, new_label)

    last_cursor_coord = coordinates
    yield

    layer._block_saving = True
    # on move
    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        interp_coord = interpolate_coordinates(
            last_cursor_coord, coordinates, layer.brush_size
        )
        for c in interp_coord:
            if layer._mode in [Mode.PAINT, Mode.ERASE]:
                layer.paint(c, new_label, refresh=False)
            elif layer._mode == Mode.FILL:
                layer.fill(c, new_label, refresh=False)
        layer.refresh()
        last_cursor_coord = coordinates
        yield

    # on release
    layer._block_saving = False


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = layer.get_value(event.position, world=True) or 0

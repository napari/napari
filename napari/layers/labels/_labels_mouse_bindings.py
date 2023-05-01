from napari.layers.labels._labels_constants import Mode
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate


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
    print(type(event))
    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if layer._mode == Mode.ERASE:
        new_label = layer._background_label
    else:
        new_label = layer.selected_label

    # on press
    with layer.block_history():
        layer._draw(new_label, coordinates, coordinates)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.type == 'mouse_move':
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                layer._draw(new_label, last_cursor_coord, coordinates)
            last_cursor_coord = coordinates
            yield


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = (
        layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        or 0
    )

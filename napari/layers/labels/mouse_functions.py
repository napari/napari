from copy import copy
from .labels import Labels
from .labels_utils import interpolate_coordinates
from ._constants import Mode


@Labels.mouse_drag_callbacks.append
def mouse_function(layer, event):
    """Called whenever mouse pressed in canvas.

    Parameters
    ----------
    layer : napari.layers.Labels
        Labels layer that mouse function is applied too.
    event : Event
        Vispy mouse event.
    """
    # on press
    if layer._mode == Mode.PAN_ZOOM:
        # If in pan/zoom mode do nothing
        pass
    elif layer._mode == Mode.PICKER:
        layer.selected_label = layer._value or 0
    elif layer._mode == Mode.PAINT:
        # Start painting with new label
        layer._save_history()
        layer._block_saving = True
        layer.paint(layer.coordinates, layer.selected_label)
        layer._last_cursor_coord = copy(layer.coordinates)
    elif layer._mode == Mode.FILL:
        # Fill clicked on region with new label
        layer.fill(layer.coordinates, layer._value, layer.selected_label)
    else:
        raise ValueError("Mode not recognized")

    yield

    # on move
    while event.type == 'mouse_move':
        if layer._mode == Mode.PAINT and event.is_dragging:
            if layer._last_cursor_coord is None:
                interp_coord = [layer.coordinates]
            else:
                interp_coord = interpolate_coordinates(
                    layer._last_cursor_coord,
                    layer.coordinates,
                    layer.brush_size,
                )
            for c in interp_coord:
                layer.paint(c, layer.selected_label, refresh=False)
            layer.refresh()
            layer._last_cursor_coord = copy(layer.coordinates)
        yield

    # on release
    layer._last_cursor_coord = None
    layer._block_saving = False

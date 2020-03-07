import numpy as np

from ._points_utils import points_in_box


def select(layer, event):
    """Select points."""
    # on press
    shift = 'Shift' in event.modifiers
    if shift and layer._value is not None:
        if layer._value in layer.selected_data:
            layer.selected_data = [
                x for x in layer.selected_data if x != layer._value
            ]
        else:
            layer.selected_data += [layer._value]
    elif layer._value is not None:
        if layer._value not in layer.selected_data:
            layer.selected_data = [layer._value]
    else:
        layer.selected_data = []
    layer._set_highlight()
    yield

    # on move
    while event.type == 'mouse_move':
        if len(layer.selected_data) > 0:
            layer._move(layer.selected_data, layer.coordinates)
        else:
            layer._is_selecting = True
            if layer._drag_start is None:
                layer._drag_start = [
                    layer.coordinates[d] for d in layer.dims.displayed
                ]
            layer._drag_box = np.array(
                [
                    layer._drag_start,
                    [layer.coordinates[d] for d in layer.dims.displayed],
                ]
            )
            layer._set_highlight()
        yield

    # on release
    layer._drag_start = None
    if layer._is_selecting:
        layer._is_selecting = False
        if len(layer._view_data) > 0:
            selection = points_in_box(
                layer._drag_box, layer._view_data, layer._view_size
            )
            layer.selected_data = layer._indices_view[selection]
        else:
            layer.selected_data = []
        layer._set_highlight(force=True)


def add(layer, event):
    """Add a point at the cursor position."""
    # on press
    layer.add(layer.coordinates)
    yield

    # on move
    while event.type == 'mouse_move':
        yield

    # on release
    layer._drag_start = None
    if layer._is_selecting:
        layer._is_selecting = False
        if len(layer._view_data) > 0:
            selection = points_in_box(
                layer._drag_box, layer._view_data, layer._view_size
            )
            layer.selected_data = layer._indices_view[selection]
        else:
            layer.selected_data = []
        layer._set_highlight(force=True)


def highlight(layer, event):
    """Highlight hovered points."""
    layer._set_highlight()

import numpy as np

from ._points_utils import points_in_box


def select(layer, event):
    """Select points.

    Clicking on a point will select that point. If holding shift while clicking
    that point will be added to or removed from the existing selection
    depending on whether it is selected or not.

    Clicking and dragging a point that is already selected will drag all the
    currently selected points.

    Clicking and dragging on an empty part of the canvas (i.e. not on a point)
    will create a drag box that will select all points inside it when finished.
    Holding shift throughout the entirety of this process will add those points
    to any existing selection, otherwise these will become the only selected
    points.
    """
    # on press
    shift = 'Shift' in event.modifiers

    # if shift add / remove any from existing selection
    if shift:
        if layer._value is not None:
            if layer._value in layer.selected_data:
                layer.selected_data = [
                    x for x in layer.selected_data if x != layer._value
                ]
            else:
                layer.selected_data += [layer._value]
        else:
            pass
    else:
        if layer._value is not None:
            if layer._value not in layer.selected_data:
                layer.selected_data = [layer._value]
        else:
            layer.selected_data = []
    layer._set_highlight()
    yield

    # on move
    while event.type == 'mouse_move':
        # If not holding shift and some points selected then drag them
        if not shift and len(layer.selected_data) > 0:
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
            # If shift combine drag selection with existing selected ones
            if shift:
                new_selected = layer._indices_view[selection]
                target = set(layer.selected_data).symmetric_difference(
                    set(new_selected)
                )
                layer.selected_data = list(target)
            else:
                layer.selected_data = layer._indices_view[selection]
        else:
            layer.selected_data = []
    layer._set_highlight(force=True)


def add(layer, event):
    """Add a new point at the clicked position."""
    # on press
    layer.add(layer.coordinates)


def highlight(layer, event):
    """Highlight hovered points."""
    layer._set_highlight()

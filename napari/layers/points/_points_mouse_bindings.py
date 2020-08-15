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
    modify_selection = (
        'Shift' in event.modifiers or 'Control' in event.modifiers
    )

    # if modifying selection add / remove any from existing selection
    if modify_selection:
        # layer._value is defined in the base layer and contains the value
        # under the cursor. For points, this is the index of the highlighted
        # point.
        if layer._value is not None:
            layer.selected_data = _toggle_selected(
                layer.selected_data, layer._value
            )
    else:
        if layer._value is not None:
            # If the current index is not in the current list make it the only
            # index selected, otherwise don't change the selection so that
            # the current selection can be dragged together.
            if layer._value not in layer.selected_data:
                layer.selected_data = {layer._value}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    yield

    # on move
    while event.type == 'mouse_move':
        # If not holding modifying selection and points selected then drag them
        if not modify_selection and len(layer.selected_data) > 0:
            layer._move(layer.selected_data, layer.coordinates)
        else:
            layer._is_selecting = True
            if layer._drag_start is None:
                layer._drag_start = layer.displayed_coordinates
            layer._drag_box = np.array(
                [layer._drag_start, layer.displayed_coordinates]
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
            if modify_selection:
                new_selected = layer._indices_view[selection]
                target = set(layer.selected_data).symmetric_difference(
                    set(new_selected)
                )
                layer.selected_data = list(target)
            else:
                layer.selected_data = layer._indices_view[selection]
        else:
            layer.selected_data = set()
    layer._set_highlight(force=True)


def add(layer, event):
    """Add a new point at the clicked position."""
    # on press
    layer.add(layer.coordinates)


def highlight(layer, event):
    """Highlight hovered points."""
    layer._set_highlight()


def _toggle_selected(selected_data, value):
    """Add or remove value from the selected data set.

    Parameters
    ----------
    selected_data : set
        Set of selected data points to be modified.
    value : int
        Index of point to add or remove from selected data set.

    Returns
    -------
    set
        Modified selected_data set.
    """
    if value in selected_data:
        selected_data.remove(value)
    else:
        selected_data.add(value)

    return selected_data

from typing import Set, TypeVar

import numpy as np

from napari.layers.base import ActionType
from napari.layers.points._points_utils import _points_in_box_3d, points_in_box


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

    # Get value under the cursor, for points, this is the index of the highlighted
    # if any, or None.
    value = layer.get_value(
        position=event.position,
        view_direction=event.view_direction,
        dims_displayed=event.dims_displayed,
        world=True,
    )
    # if modifying selection add / remove any from existing selection
    if modify_selection:
        if value is not None:
            layer.selected_data = _toggle_selected(layer.selected_data, value)
    else:
        if value is not None:
            # If the current index is not in the current list make it the only
            # index selected, otherwise don't change the selection so that
            # the current selection can be dragged together.
            if value not in layer.selected_data:
                layer.selected_data = {value}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    # Set _drag_start value here to prevent an offset when mouse_move happens
    # https://github.com/napari/napari/pull/4999
    layer._set_drag_start(
        layer.selected_data,
        layer.world_to_data(event.position),
        center_by_data=not modify_selection,
    )
    yield

    # Undo the toggle selected in case of a mouse move with modifiers
    if modify_selection and value is not None and event.type == 'mouse_move':
        layer.selected_data = _toggle_selected(layer.selected_data, value)

    is_moving = False
    # on move
    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        # If not holding modifying selection and points selected then drag them
        if not modify_selection and len(layer.selected_data) > 0:
            # only emit just before moving
            if not is_moving:
                layer.events.data(
                    value=layer.data,
                    action=ActionType.CHANGING,
                    data_indices=tuple(
                        layer.selected_data,
                    ),
                    vertex_indices=((),),
                )
            is_moving = True
            with layer.events.data.blocker():
                layer._move(layer.selected_data, coordinates)
        else:
            # while dragging, update the drag box
            coord = [coordinates[i] for i in layer._slice_input.displayed]
            layer._is_selecting = True
            layer._drag_box = np.array([layer._drag_start, coord])

            # update the drag up and normal vectors on the layer
            _update_drag_vectors_from_event(layer=layer, event=event)

            layer._set_highlight()
        yield

    # only emit data once dragging has finished
    if is_moving:
        layer._move([], coordinates)
        is_moving = False

    # on release
    layer._drag_start = None
    if layer._is_selecting:
        # if drag selection was being performed, select points
        # using the drag box
        layer._is_selecting = False
        n_display = len(event.dims_displayed)
        _select_points_from_drag(
            layer=layer, modify_selection=modify_selection, n_display=n_display
        )

    # reset the selection box data and highlights
    layer._drag_box = None
    layer._drag_normal = None
    layer._drag_up = None
    layer._set_highlight(force=True)


DRAG_DIST_THRESHOLD = 5


def add(layer, event):
    """Add a new point at the clicked position."""
    start_pos = event.pos
    dist = 0
    yield

    while event.type == 'mouse_move':
        dist = np.linalg.norm(start_pos - event.pos)
        if dist < DRAG_DIST_THRESHOLD:
            # prevent vispy from moving the canvas if we're below threshold
            event.handled = True
        yield

    # in some weird cases you might have press and release without move,
    # so we just make 100% sure dist is correct
    dist = np.linalg.norm(start_pos - event.pos)
    if dist < DRAG_DIST_THRESHOLD:
        coordinates = layer.world_to_data(event.position)
        layer.add(coordinates)


def highlight(layer, event):
    """Highlight hovered points."""
    layer._set_highlight()


_T = TypeVar("_T")


def _toggle_selected(selection: Set[_T], value: _T) -> Set[_T]:
    """Add or remove value from the selection set.

    This function returns a copy of the existing selection.

    Parameters
    ----------
    selection: set
        Set of selected data points to be modified.
    value : int
        Index of point to add or remove from selected data set.

    Returns
    -------
    selection: set
        Updated selection.
    """
    selection = set(selection)
    if value in selection:
        selection.remove(value)
    else:
        selection.add(value)
    return selection


def _update_drag_vectors_from_event(layer, event):
    """Update the drag normal and up vectors on layer from a mouse event.

    Note that in 2D mode, the layer._drag_normal and layer._drag_up
    are set to None.

    Parameters
    ----------
    layer : "napari.layers.Points"
        The Points layer to update.
    event
        The mouse event object.
    """
    n_display = len(event.dims_displayed)
    if n_display == 3:
        # if in 3D, set the drag normal and up directions
        # get the indices of the displayed dimensions
        ndim_world = len(event.position)
        layer_dims_displayed = layer._world_to_layer_dims(
            world_dims=event.dims_displayed, ndim_world=ndim_world
        )

        # get the view direction in displayed data coordinates
        layer._drag_normal = layer._world_to_displayed_data_ray(
            event.view_direction, layer_dims_displayed
        )

        # get the up direction of the camera in displayed data coordinates
        layer._drag_up = layer._world_to_displayed_data_ray(
            event.up_direction, layer_dims_displayed
        )

    else:
        # if in 2D, set the drag normal and up to None
        layer._drag_normal = None
        layer._drag_up = None


def _select_points_from_drag(layer, modify_selection: bool, n_display: int):
    """Select points on a Points layer after a drag event.

    Parameters
    ----------
    layer : napari.layers.Points
        The points layer to select points on.
    modify_selection : bool
        Set to true if the selection should modify the current selected data
        in layer.selected_data.
    n_display : int
        The number of dimensions current being displayed
    """
    if len(layer._view_data) == 0:
        # if no data in view, there isn't any data to select
        layer.selected_data = set()

    # if there is data in view, find the points in the drag box
    if n_display == 2:
        selection = points_in_box(
            layer._drag_box, layer._view_data, layer._view_size
        )
    else:
        selection = _points_in_box_3d(
            layer._drag_box,
            layer._view_data,
            layer._view_size,
            layer._drag_normal,
            layer._drag_up,
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

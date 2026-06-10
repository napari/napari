from __future__ import annotations

import warnings
from collections.abc import Collection
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from napari.layers.base import ActionType
from napari.layers.base._base_constants import InteractionBoxHandle
from napari.layers.base._base_mouse_bindings import (
    highlight_selection_box_handles,
)
from napari.layers.points._points_utils import points_in_box
from napari.layers.utils.interaction_box import (
    generate_interaction_box_handles,
)
from napari.layers.utils.layer_utils import dims_displayed_world_to_layer
from napari.utils.events import Event

if TYPE_CHECKING:
    from collections.abc import Generator, Set as AbstractSet

    from napari.layers.points.points import Points


def select(layer: Points, event: Event) -> Generator[None, None, None]:
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

    Holding Ctrl will show a transformable box around the selected points,
    allowing to resize/move/rotate the selection using the respective handles.
    """
    start_pos_canvas = event.pos[::-1]
    start_pos_world = np.array(event.position)

    if 'Control' in event.modifiers:
        yield from _transform_selection_box(layer, event, start_pos_world)
        return

    modify_selection = 'Shift' in event.modifiers

    # Get value under the cursor, for points, this is the index of the highlighted
    # if any, or None.
    value = layer._get_value_(
        position=event.position,
        view_direction=event.view_direction,
        dims_displayed=event.dims_displayed,
        world=True,
    )

    old_selection = set(layer.selected_data)
    new_selection = None
    if value is not None:
        # if modifying selection add / remove any from existing selection
        if modify_selection:
            new_selection = _toggle_selected(layer.selected_data, value)
        else:
            # If the current index is not in the current list make it the only
            # index selected, otherwise don't change the selection so that
            # the current selection can be dragged together.
            if value not in layer.selected_data:
                new_selection = {value}
    elif not modify_selection:
        new_selection = {}

    yield

    if event.type == 'mouse_release':
        # it was an actual click (not drag), so we can toggle the selection
        if new_selection is not None:
            layer.selected_data = new_selection
        return

    # the following code only happens on *drag*

    # only move points if clicking on a selected point and if we're not modifying
    if value in old_selection and not modify_selection:
        yield from _move_selection(layer, event, start_pos_world)
    else:
        yield from _select_with_rectangle(
            layer,
            event,
            start_pos_canvas,
            start_pos_world,
            modify_selection,
        )


def _select_with_rectangle(
    layer: Points,
    event: Event,
    start_pos_canvas: np.ndarray,
    start_pos_world: np.ndarray,
    modify_selection: bool,
) -> Generator[None, None, None]:
    rect = layer._overlays['selection_rect']
    rect.corners_canvas = (start_pos_canvas, start_pos_canvas)
    rect.corners_world = (start_pos_world, start_pos_world)
    rect.visible = True

    initial_selection = set(layer.selected_data)

    while event.type == 'mouse_move':
        pos_canvas = event.pos[::-1]
        pos_world = event.position
        rect.corners_canvas = (start_pos_canvas, pos_canvas)
        rect.corners_world = (start_pos_world, pos_world)

        corners = np.array(rect.corners_world)
        corners_data = np.array(
            [layer.world_to_data(corners[0]), layer.world_to_data(corners[1])]
        )
        displayed = dims_displayed_world_to_layer(
            dims_displayed_world=event.dims_displayed,
            ndim_world=len(event.position),
            ndim_layer=layer.ndim,
        )

        # TODO: this is broken by changing order, why?
        #       I think we need to index _view_data with "layer order",
        #       but not "displayed" because _view_data already is down in
        #       dimensions based on which are displayed (but not order?)
        selected = set(
            points_in_box(
                corners_data[:, displayed],
                layer._view_data,
                layer._view_size,
            )
        )
        if modify_selection:
            selected = _toggle_selected(initial_selection, selected)
        layer.selected_data = selected
        layer.events.highlight()
        yield

    rect.visible = False


def _move_selection(
    layer: Points, event: Event, start_pos
) -> Generator[None, None, None]:
    # do everything relative to the original data to remove drift
    selected = np.fromiter(layer.selected_data, dtype=int)
    data_orig = layer.data[selected].copy()

    while event.type == 'mouse_move':
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGING,
            data_indices=selected,
            vertex_indices=((),),
        )
        pos = np.array(event.position)
        shift = layer.world_to_data(pos - start_pos)
        layer.data[selected] = data_orig + shift
        layer.refresh()
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=selected,
            vertex_indices=((),),
        )
        layer.events.features()
        box = layer._overlays['selection_box']
        # TODO: this needs to work with any dims and order, currently hardcoded.
        #       Probably also shouldn't be selected_view
        box.update_from_points(layer.data[layer._selected_view][:, -2:])
        yield


def _resize_selection(
    layer: Points,
    event: Event,
    start_pos,
    dragged_handle,
) -> Generator[None, None, None]:
    box = layer._overlays['selection_box']
    fixed_handle = InteractionBoxHandle.opposite_handle(dragged_handle)

    # get exact coordinates of original handle to avoid unwanted shifts
    handle_coords = generate_interaction_box_handles(*box.bounds)
    fixed_handle_coords = handle_coords[fixed_handle]
    dragged_handle_coords = handle_coords[dragged_handle]
    handles_vector = dragged_handle_coords - fixed_handle_coords

    selected = tuple(layer.selected_data)
    selected_displayed = np.ix_(selected, event.dims_displayed)
    data_orig = layer.data[selected_displayed].copy()

    while event.type == 'mouse_move':
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGING,
            data_indices=selected,
            vertex_indices=((),),
        )
        pos = np.array(event.position)[event.dims_displayed]
        shift = pos - start_pos
        with warnings.catch_warnings():
            # a "divide by zero" warning is raised here when resizing along only one axis
            # (i.e: dragging the central handle of the Box).
            # That's intended, because we get inf or nan, which we can then replace with 1s
            # and thus maintain the size along that axis.
            warnings.simplefilter('ignore', RuntimeWarning)
            scale = (handles_vector + shift) / handles_vector
            scale = np.nan_to_num(scale, posinf=1, neginf=1, nan=1)
        layer.data[selected_displayed] = (
            fixed_handle_coords + (data_orig - fixed_handle_coords) * scale
        )
        layer.refresh()
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=selected,
            vertex_indices=((),),
        )
        layer.events.features()
        selected = layer.data[np.fromiter(layer.selected_data, dtype=int)]
        box.update_from_points(selected)
        yield


def _rotate_selection(
    layer: Points,
    event: Event,
    start_pos,
    dragged_handle,
) -> Generator[None, None, None]:
    while event.type == 'mouse_move':
        yield


def _transform_selection_box(
    layer: Points, event: Event, start_pos
) -> Generator[None, None, None]:
    box = layer._overlays['selection_box']
    if len(event.dims_displayed) != 2:
        return

    clicked_handle = box.selected_handle

    if clicked_handle is None:
        return

    yield

    if clicked_handle == InteractionBoxHandle.INSIDE:
        yield from _move_selection(layer, event, start_pos)
    elif clicked_handle == InteractionBoxHandle.ROTATION:
        yield from _rotate_selection(layer, event, start_pos, clicked_handle)
    else:
        yield from _resize_selection(layer, event, start_pos, clicked_handle)


DRAG_DIST_THRESHOLD = 5


def add(layer: Points, event: Event) -> Generator[None, None, None]:
    """Add a new point at the clicked position."""
    start_pos = event.pos
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


def highlight(layer: Points, event: Event) -> None:
    """Highlight hovered points."""
    box = layer._overlays['selection_box']
    # TODO: how to make this appear/disappear on clicking shift
    #       instead of on mouse move? Separate key binding just on shift?
    box.visible = False
    box.handles = False
    box.selected_handle = None

    if 'Control' in event.modifiers and len(layer.selected_data) > 1:
        box.handles = True
        box.visible = True
        displayed = dims_displayed_world_to_layer(
            dims_displayed_world=event.dims_displayed,
            ndim_world=len(event.position),
            ndim_layer=layer.ndim,
        )
        selected = np.fromiter(layer.selected_data, dtype=int)
        box.update_from_points(layer.data[selected][:, displayed])
        highlight_selection_box_handles(layer, event)
    else:
        value = layer._get_value_(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if value is None:
            layer._highlight_index = []
        else:
            layer._highlight_index = [value]
        layer.events.highlight()


_T = TypeVar('_T')


def _toggle_selected(
    selection: AbstractSet[_T], to_toggle: _T | Collection[_T]
) -> set[_T]:
    """Add or remove value from the selection set.

    Parameters
    ----------
    selection : set
        Set of selected data points to be modified.
    value : int or Collection
        Index of point to add or remove from selected data set.

    Returns
    -------
    selection: set
        Updated selection.
    """
    selection = set(selection)
    to_toggle = (
        {to_toggle}
        if not isinstance(to_toggle, Collection)
        else set(to_toggle)
    )
    selection.symmetric_difference_update(to_toggle)
    return selection

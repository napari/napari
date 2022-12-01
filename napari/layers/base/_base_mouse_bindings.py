import numpy as np

from napari.components.overlays._interaction_box_constants import (
    InteractionBoxHandle,
)
from napari.utils.geometry import generate_interaction_box_vertices


def _get_interaction_box_position(box_bounds, event):
    # generates in vispy canvas coordinates, so invert x and y
    top_left, bot_right = (tuple(point) for point in box_bounds.T[:, ::-1])
    vertices = generate_interaction_box_vertices(
        top_left, bot_right, handles=True
    )[:, ::-1]

    pos = np.array(event.position)[event.dims_displayed]
    dist = np.linalg.norm(pos - vertices, axis=1)
    tolerance = dist.max() / 100
    close_to_vertex = np.isclose(dist, 0, atol=tolerance)
    if np.any(close_to_vertex):
        return InteractionBoxHandle(np.argmax(close_to_vertex))
    elif np.all((pos[::-1] >= top_left) & (pos[::-1] <= bot_right)):
        return InteractionBoxHandle.ALL
    else:
        return None


def highlight_box_handles(layer, event):
    if not layer._overlays['transform_box'].visible:
        return

    bounds = layer._display_bounding_box(event.dims_displayed)
    nearby_handle = _get_interaction_box_position(bounds, event)
    if nearby_handle is not None:
        layer._overlays['transform_box'].selected_vertex = nearby_handle


def transform_with_box(layer, event):
    if not layer._overlays['transform_box'].visible:
        return

    bounds = layer._display_bounding_box(event.dims_displayed)
    nearby_handle = _get_interaction_box_position(bounds, event)

    if nearby_handle is not None:
        print(f'picked {repr(nearby_handle)}')

    # modify_selection = (
    #     'Shift' in event.modifiers or 'Control' in event.modifiers
    # )
    # # Get value under the cursor, for points, this is the index of the highlighted
    # # if any, or None.
    # value = layer.get_value(
    #     position=event.position,
    #     view_direction=event.view_direction,
    #     dims_displayed=event.dims_displayed,
    #     world=True,
    # )
    # # if modifying selection add / remove any from existing selection
    # if modify_selection:
    #     if value is not None:
    #         layer.selected_data = _toggle_selected(layer.selected_data, value)
    # else:
    #     if value is not None:
    #         # If the current index is not in the current list make it the only
    #         # index selected, otherwise don't change the selection so that
    #         # the current selection can be dragged together.
    #         if value not in layer.selected_data:
    #             layer.selected_data = {value}
    #     else:
    #         layer.selected_data = set()
    # layer._set_highlight()
    #
    # # Set _drag_start value here to prevent an offset when mouse_move happens
    # # https://github.com/napari/napari/pull/4999
    # layer._set_drag_start(
    #     layer.selected_data,
    #     layer.world_to_data(event.position),
    #     center_by_data=not modify_selection,
    # )
    # yield
    #
    # # Undo the toggle selected in case of a mouse move with modifiers
    # if modify_selection and value is not None and event.type == 'mouse_move':
    #     layer.selected_data = _toggle_selected(layer.selected_data, value)
    #
    # is_moving = False
    # # on move
    # while event.type == 'mouse_move':
    #     coordinates = layer.world_to_data(event.position)
    #     # If not holding modifying selection and points selected then drag them
    #     if not modify_selection and len(layer.selected_data) > 0:
    #         is_moving = True
    #         with layer.events.data.blocker():
    #             layer._move(layer.selected_data, coordinates)
    #     else:
    #         # while dragging, update the drag box
    #         coord = [coordinates[i] for i in layer._slice_input.displayed]
    #         layer._is_selecting = True
    #         layer._drag_box = np.array([layer._drag_start, coord])
    #
    #         # update the drag up and normal vectors on the layer
    #         _update_drag_vectors_from_event(layer=layer, event=event)
    #
    #         layer._set_highlight()
    #     yield
    #
    # # only emit data once dragging has finished
    # if is_moving:
    #     layer._move([], coordinates)
    #     is_moving = False
    #
    # # on release
    # layer._drag_start = None
    # if layer._is_selecting:
    #     # if drag selection was being performed, select points
    #     # using the drag box
    #     layer._is_selecting = False
    #     n_display = len(event.dims_displayed)
    #     _select_points_from_drag(
    #         layer=layer, modify_selection=modify_selection, n_display=n_display
    #     )
    #
    # # reset the selection box data and highlights
    # layer._drag_box = None
    # layer._drag_normal = None
    # layer._drag_up = None
    # layer._set_highlight(force=True)
    #

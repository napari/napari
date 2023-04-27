from __future__ import annotations

from copy import copy

import numpy as np

from napari.layers.shapes._shapes_constants import Box, Mode
from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.layers.shapes._shapes_utils import point_to_lines, rdp


def highlight(layer, event):
    """Highlight hovered shapes."""
    layer._set_highlight()


def select(layer, event):
    """Select shapes or vertices either in select or direct select mode.

    Once selected shapes can be moved or resized, and vertices can be moved
    depending on the mode. Holding shift when resizing a shape will preserve
    the aspect ratio.
    """
    shift = 'Shift' in event.modifiers
    # on press
    value = layer.get_value(event.position, world=True)
    layer._moving_value = copy(value)
    shape_under_cursor, vertex_under_cursor = value
    if vertex_under_cursor is None:
        if shift and shape_under_cursor is not None:
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            else:
                if len(layer.selected_data):
                    # one or more shapes already selected
                    layer.selected_data.add(shape_under_cursor)
                else:
                    # first shape being selected
                    layer.selected_data = {shape_under_cursor}
        elif shape_under_cursor is not None:
            if shape_under_cursor not in layer.selected_data:
                layer.selected_data = {shape_under_cursor}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    # we don't update the thumbnail unless a shape has been moved
    update_thumbnail = False

    # Set _drag_start value here to prevent an offset when mouse_move happens
    # https://github.com/napari/napari/pull/4999
    _set_drag_start(layer, layer.world_to_data(event.position))
    yield

    # on move
    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        # ToDo: Need to pass moving_coordinates to allow fixed aspect ratio
        # keybinding to work, this should be dropped
        layer._moving_coordinates = coordinates
        # Drag any selected shapes
        if len(layer.selected_data) == 0:
            _drag_selection_box(layer, coordinates)
        else:
            _move(layer, coordinates)

        # if a shape is being moved, update the thumbnail
        if layer._is_moving:
            update_thumbnail = True
        yield

    # only emit data once dragging has finished
    if layer._is_moving:
        layer.events.data(value=layer.data)

    # on release
    shift = 'Shift' in event.modifiers
    if not layer._is_moving and not layer._is_selecting and not shift:
        if shape_under_cursor is not None:
            layer.selected_data = {shape_under_cursor}
        else:
            layer.selected_data = set()
    elif layer._is_selecting:
        layer.selected_data = layer._data_view.shapes_in_box(layer._drag_box)
        layer._is_selecting = False
        layer._set_highlight()

    layer._is_moving = False
    layer._drag_start = None
    layer._drag_box = None
    layer._fixed_vertex = None
    layer._moving_value = (None, None)
    layer._set_highlight()

    if update_thumbnail:
        layer._update_thumbnail()


def add_line(layer, event):
    """Add a line."""
    size = layer._vertex_size * layer.scale_factor / 4
    full_size = np.zeros(layer.ndim, dtype=float)
    for i in layer._slice_input.displayed:
        full_size[i] = size

    coordinates = layer.world_to_data(event.position)
    layer._moving_coordinates = coordinates

    corner = np.array(coordinates)
    data = np.array([corner, corner + full_size])
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='line'
    )


def add_ellipse(layer, event):
    """Add an ellipse."""
    size = layer._vertex_size * layer.scale_factor / 4
    size_h = np.zeros(layer.ndim, dtype=float)
    size_h[layer._slice_input.displayed[0]] = size
    size_v = np.zeros(layer.ndim, dtype=float)
    size_v[layer._slice_input.displayed[1]] = size

    coordinates = layer.world_to_data(event.position)
    corner = np.array(coordinates)
    data = np.array(
        [corner, corner + size_v, corner + size_h + size_v, corner + size_h]
    )
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='ellipse'
    )


def add_rectangle(layer, event):
    """Add a rectangle."""
    size = layer._vertex_size * layer.scale_factor / 4
    size_h = np.zeros(layer.ndim, dtype=float)
    size_h[layer._slice_input.displayed[0]] = size
    size_v = np.zeros(layer.ndim, dtype=float)
    size_v[layer._slice_input.displayed[1]] = size

    coordinates = layer.world_to_data(event.position)
    corner = np.array(coordinates)
    data = np.array(
        [corner, corner + size_v, corner + size_h + size_v, corner + size_h]
    )

    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='rectangle'
    )


def _add_line_rectangle_ellipse(layer, event, data, shape_type):
    """Helper function for adding a line, rectangle or ellipse."""

    # on press
    # Start drawing rectangle / ellipse / line
    layer.add(data, shape_type=shape_type)
    layer.selected_data = {layer.nshapes - 1}
    layer._value = (layer.nshapes - 1, 4)
    layer._moving_value = copy(layer._value)
    layer.refresh()
    yield

    # on move
    while event.type == 'mouse_move':
        # Drag any selected shapes
        coordinates = layer.world_to_data(event.position)
        layer._moving_coordinates = coordinates
        _move(layer, coordinates)
        yield

    # on release
    layer._finish_drawing()


def finish_drawing_shape(layer, event):
    """
    finish drawing the current shape
    """
    layer._finish_drawing()


def add_path_polygon_tablet(layer, event):
    """Creating, drawing and finishing the polygon shape while in tablet mode. Reason for separating from
    add_path_polygon is that a yield is required which turns the whole function into a generator even when the yield
    is not reached. This breaks the mouse draw polygon functionality."""
    # on press
    coordinates = layer.world_to_data(event.position)

    if layer._is_creating is False:
        # Reset last cursor position in case shapes were drawn in different dimension beforehand.
        global _last_cursor_position
        _last_cursor_position = None

        # Start drawing a path
        data = np.array([coordinates, coordinates])
        layer.add(data, shape_type='path')
        layer.selected_data = {layer.nshapes - 1}
        layer._value = (layer.nshapes - 1, 1)
        layer._moving_value = copy(layer._value)
        layer._is_creating = True
        layer._set_highlight()
        if layer._mode == Mode.ADD_POLYGON_LASSO_TABLET:
            yield
            while event.type == 'mouse_move':
                # TODO add functionality of adding datapoints only if distance threshold is met. Currently not respected.
                add_path_polygon_lasso_creating(layer, event)
                index = layer._moving_value[0]
                new_type = Polygon
                vertices = layer._data_view.shapes[index].data
                vertices = np.concatenate((vertices, [coordinates]), axis=0)
                # Change the selected vertex
                value = layer.get_value(event.position, world=True)
                layer._value = (value[0], value[1] + 1)
                layer._moving_value = copy(layer._value)
                layer._data_view.edit(index, vertices, new_type=new_type)
                layer._selected_box = layer.interaction_box(
                    layer.selected_data
                )
                yield
            index = layer._moving_value[0]
            vertices = layer._data_view.shapes[index].data
            vertices = rdp(vertices, epsilon=0.5)
            layer._data_view.edit(index, vertices, new_type=Polygon)
            finish_drawing_shape(layer, event)


def add_path_polygon(layer, event):
    """Add a path or polygon."""
    # on press
    coordinates = layer.world_to_data(event.position)
    if layer._is_creating is False:
        # Reset last cursor position in case shapes were drawn in different dimension beforehand.
        global _last_cursor_position
        _last_cursor_position = None

        # Start drawing a path
        data = np.array([coordinates, coordinates])
        layer.add(data, shape_type='path')
        layer.selected_data = {layer.nshapes - 1}
        layer._value = (layer.nshapes - 1, 1)
        layer._moving_value = copy(layer._value)
        layer._is_creating = True
        layer._set_highlight()
    elif event.type == 'mouse_press' and layer._mode == Mode.ADD_POLYGON_LASSO:
        index = layer._moving_value[0]
        vertices = layer._data_view.shapes[index].data
        vertices = rdp(vertices, epsilon=0.5)
        layer._data_view.edit(index, vertices, new_type=Polygon)
        finish_drawing_shape(layer, event)
    else:
        # Add to an existing path or polygon
        index = layer._moving_value[0]
        if layer._mode in {
            Mode.ADD_POLYGON,
            Mode.ADD_POLYGON_LASSO,
        }:
            new_type = Polygon
        else:
            new_type = None

        vertices = layer._data_view.shapes[index].data
        vertices = np.concatenate((vertices, [coordinates]), axis=0)
        # Change the selected vertex
        value = layer.get_value(event.position, world=True)
        layer._value = (value[0], value[1] + 1)
        layer._moving_value = copy(layer._value)
        layer._data_view.edit(index, vertices, new_type=new_type)
        layer._selected_box = layer.interaction_box(layer.selected_data)


def add_path_polygon_creating(layer, event):
    """While a path or polygon move next vertex to be added."""
    if layer._is_creating:
        coordinates = layer.world_to_data(event.position)
        _move(layer, coordinates)


def add_path_polygon_lasso_creating(layer, event):
    """While a path or polygon move next vertex to be added."""
    # print(f'add_path_polygon_lasso_creating(): layer._is_creating = {layer._is_creating}, event.type = {event.type}, event.is_dragging = {event.is_dragging}')
    if layer._is_creating:
        coordinates = layer.world_to_data(event.position)
        _move(layer, coordinates)

        global _last_cursor_position
        if _last_cursor_position is not None:
            position_diff = np.linalg.norm(event.pos - _last_cursor_position)
            if position_diff < 10:
                return
        # Use screen position instead of world / data position to account for zoom
        _last_cursor_position = np.array(event.pos)
        add_path_polygon(layer, event)


def vertex_insert(layer, event):
    """Insert a vertex into a selected shape.

    The vertex will get inserted in between the vertices of the closest edge
    from all the edges in selected shapes. Vertices cannot be inserted into
    Ellipses.
    """
    # Determine all the edges in currently selected shapes
    all_edges = np.empty((0, 2, 2))
    all_edges_shape = np.empty((0, 2), dtype=int)
    for index in layer.selected_data:
        shape_type = type(layer._data_view.shapes[index])
        if shape_type == Ellipse:
            # Adding vertex to ellipse not implemented
            pass
        else:
            vertices = layer._data_view.displayed_vertices[
                layer._data_view.displayed_index == index
            ]
            # Find which edge new vertex should inserted along
            closed = shape_type != Path
            n = len(vertices)
            if closed:
                lines = np.array(
                    [[vertices[i], vertices[(i + 1) % n]] for i in range(n)]
                )
            else:
                lines = np.array(
                    [[vertices[i], vertices[i + 1]] for i in range(n - 1)]
                )
            all_edges = np.append(all_edges, lines, axis=0)
            indices = np.array(
                [np.repeat(index, len(lines)), list(range(len(lines)))]
            ).T
            all_edges_shape = np.append(all_edges_shape, indices, axis=0)

    if len(all_edges) == 0:
        # No appropriate edges were found
        return

    # Determine the closet edge to the current cursor coordinate
    coordinates = layer.world_to_data(event.position)
    coord = [coordinates[i] for i in layer._slice_input.displayed]
    ind, loc = point_to_lines(coord, all_edges)
    index = all_edges_shape[ind][0]
    ind = all_edges_shape[ind][1] + 1
    shape_type = type(layer._data_view.shapes[index])
    if shape_type == Line:
        # Adding vertex to line turns it into a path
        new_type = Path
    elif shape_type == Rectangle:
        # Adding vertex to rectangle turns it into a polygon
        new_type = Polygon
    else:
        new_type = None
    closed = shape_type != Path
    vertices = layer._data_view.shapes[index].data
    if not closed:
        if int(ind) == 1 and loc < 0:
            ind = 0
        elif int(ind) == len(vertices) - 1 and loc > 1:
            ind = ind + 1

    # Insert new vertex at appropriate place in vertices of target shape
    vertices = np.insert(vertices, ind, [coordinates], axis=0)
    with layer.events.set_data.blocker():
        layer._data_view.edit(index, vertices, new_type=new_type)
        layer._selected_box = layer.interaction_box(layer.selected_data)
    layer.refresh()


def vertex_remove(layer, event):
    """Remove a vertex from a selected shape.

    If a vertex is clicked on remove it from the shape it is in. If this cause
    the shape to shrink to a size that no longer is valid remove the whole
    shape.
    """
    value = layer.get_value(event.position, world=True)
    shape_under_cursor, vertex_under_cursor = value
    if vertex_under_cursor is None:
        # No vertex was clicked on so return
        return

    # Have clicked on a current vertex so remove
    shape_type = type(layer._data_view.shapes[shape_under_cursor])
    if shape_type == Ellipse:
        # Removing vertex from ellipse not implemented
        return
    vertices = layer._data_view.shapes[shape_under_cursor].data
    if len(vertices) <= 2 or (shape_type == Polygon and len(vertices) == 3):
        # If only 2 vertices present, remove whole shape
        with layer.events.set_data.blocker():
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            layer._data_view.remove(shape_under_cursor)
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    else:
        if shape_type == Rectangle:  # noqa SIM108
            # Deleting vertex from a rectangle creates a polygon
            new_type = Polygon
        else:
            new_type = None
        # Remove clicked on vertex
        vertices = np.delete(vertices, vertex_under_cursor, axis=0)
        with layer.events.set_data.blocker():
            layer._data_view.edit(
                shape_under_cursor, vertices, new_type=new_type
            )
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    layer.refresh()


def _drag_selection_box(layer, coordinates):
    """Drag a selection box.

    Parameters
    ----------
    layer : napari.layers.Shapes
        Shapes layer.
    coordinates : tuple
        Position of mouse cursor in data coordinates.
    """
    # If something selected return
    if len(layer.selected_data) > 0:
        return

    coord = [coordinates[i] for i in layer._slice_input.displayed]

    # Create or extend a selection box
    layer._is_selecting = True
    if layer._drag_start is None:
        layer._drag_start = coord
    layer._drag_box = np.array([layer._drag_start, coord])
    layer._set_highlight()


def _set_drag_start(layer, coordinates):
    coord = [coordinates[i] for i in layer._slice_input.displayed]
    if layer._drag_start is None and len(layer.selected_data) > 0:
        center = layer._selected_box[Box.CENTER]
        layer._drag_start = coord - center
    return coord


def _move(layer, coordinates):
    """Moves object at given mouse position and set of indices.

    Parameters
    ----------
    layer : napari.layers.Shapes
        Shapes layer.
    coordinates : tuple
        Position of mouse cursor in data coordinates.
    """
    # If nothing selected return
    if len(layer.selected_data) == 0:
        return

    vertex = layer._moving_value[1]

    if layer._mode in (
        [Mode.SELECT, Mode.ADD_RECTANGLE, Mode.ADD_ELLIPSE, Mode.ADD_LINE]
    ):
        coord = _set_drag_start(layer, coordinates)
        layer._moving_coordinates = coordinates
        layer._is_moving = True
        if vertex is None:
            # Check where dragging box from to move whole object
            center = layer._selected_box[Box.CENTER]
            shift = coord - center - layer._drag_start
            for index in layer.selected_data:
                layer._data_view.shift(index, shift)
            layer._selected_box = layer._selected_box + shift
            layer.refresh()
        elif vertex < Box.LEN:
            # Corner / edge vertex is being dragged so resize object
            box = layer._selected_box
            if layer._fixed_vertex is None:
                layer._fixed_index = (vertex + 4) % Box.LEN
                layer._fixed_vertex = box[layer._fixed_index]

            handle_offset = box[Box.HANDLE] - box[Box.CENTER]
            if np.linalg.norm(handle_offset) == 0:
                handle_offset = [1, 1]
            handle_offset_norm = handle_offset / np.linalg.norm(handle_offset)

            rot = np.array(
                [
                    [handle_offset_norm[0], -handle_offset_norm[1]],
                    [handle_offset_norm[1], handle_offset_norm[0]],
                ]
            )
            inv_rot = np.linalg.inv(rot)

            fixed = layer._fixed_vertex
            new = list(coord)

            c = box[Box.CENTER]
            if layer._fixed_aspect and layer._fixed_index % 2 == 0:
                # corner
                new = (box[vertex] - c) / np.linalg.norm(
                    box[vertex] - c
                ) * np.linalg.norm(new - c) + c

            if layer._fixed_index % 2 == 0:
                # corner selected
                scale = (inv_rot @ (new - fixed)) / (
                    inv_rot @ (box[vertex] - fixed)
                )
            elif layer._fixed_index % 4 == 3:
                # top or bottom selected
                scale = np.array(
                    [
                        (inv_rot @ (new - fixed))[0]
                        / (inv_rot @ (box[vertex] - fixed))[0],
                        1,
                    ]
                )
            else:
                # left or right selected
                scale = np.array(
                    [
                        1,
                        (inv_rot @ (new - fixed))[1]
                        / (inv_rot @ (box[vertex] - fixed))[1],
                    ]
                )

            # prevent box from shrinking below a threshold size
            size = [
                np.linalg.norm(box[Box.TOP_CENTER] - c),
                np.linalg.norm(box[Box.LEFT_CENTER] - c),
            ]
            threshold = layer._vertex_size * layer.scale_factor / 2
            scale[abs(scale * size) < threshold] = 1

            # check orientation of box
            if abs(handle_offset_norm[0]) == 1:
                for index in layer.selected_data:
                    layer._data_view.scale(
                        index, scale, center=layer._fixed_vertex
                    )
                layer._scale_box(scale, center=layer._fixed_vertex)
            else:
                scale_mat = np.array([[scale[0], 0], [0, scale[1]]])
                transform = rot @ scale_mat @ inv_rot
                for index in layer.selected_data:
                    layer._data_view.shift(index, -layer._fixed_vertex)
                    layer._data_view.transform(index, transform)
                    layer._data_view.shift(index, layer._fixed_vertex)
                layer._transform_box(transform, center=layer._fixed_vertex)
            layer.refresh()
        elif vertex == 8:
            # Rotation handle is being dragged so rotate object
            handle = layer._selected_box[Box.HANDLE]
            layer._fixed_vertex = layer._selected_box[Box.CENTER]
            offset = handle - layer._fixed_vertex
            layer._drag_start = -np.degrees(np.arctan2(offset[0], -offset[1]))

            new_offset = coord - layer._fixed_vertex
            new_angle = -np.degrees(np.arctan2(new_offset[0], -new_offset[1]))
            fixed_offset = handle - layer._fixed_vertex
            fixed_angle = -np.degrees(
                np.arctan2(fixed_offset[0], -fixed_offset[1])
            )

            if np.linalg.norm(new_offset) < 1:
                angle = 0
            elif layer._fixed_aspect:
                angle = np.round(new_angle / 45) * 45 - fixed_angle
            else:
                angle = new_angle - fixed_angle

            for index in layer.selected_data:
                layer._data_view.rotate(
                    index, angle, center=layer._fixed_vertex
                )
            layer._rotate_box(angle, center=layer._fixed_vertex)
            layer.refresh()

    elif (
        layer._mode
        in [
            Mode.DIRECT,
            Mode.ADD_PATH,
            Mode.ADD_POLYGON,
            Mode.ADD_POLYGON_LASSO,
            Mode.ADD_POLYGON_LASSO_TABLET,
        ]
        and vertex is not None
    ):
        layer._moving_coordinates = coordinates
        layer._is_moving = True
        index = layer._moving_value[0]
        shape_type = type(layer._data_view.shapes[index])
        if shape_type == Ellipse:
            # DIRECT vertex moving of ellipse not implemented
            pass
        else:
            new_type = Polygon if shape_type == Rectangle else None
            vertices = layer._data_view.shapes[index].data
            vertices[vertex] = coordinates
            layer._data_view.edit(index, vertices, new_type=new_type)
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
            layer.refresh()

from copy import copy

import numpy as np

from ._shapes_constants import Mode
from ._shapes_models import Ellipse, Line, Path, Polygon, Rectangle
from ._shapes_utils import point_to_lines


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
    layer._moving_value = copy(layer._value)
    shape_under_cursor, vertex_under_cursor = layer._value
    if vertex_under_cursor is None:
        if shift and shape_under_cursor is not None:
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            else:
                layer.selected_data.add(shape_under_cursor)
        elif shape_under_cursor is not None:
            if shape_under_cursor not in layer.selected_data:
                layer.selected_data = {shape_under_cursor}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    # we don't update the thumbnail unless a shape has been moved
    update_thumbnail = False
    yield

    # on move
    while event.type == 'mouse_move':
        # Drag any selected shapes
        layer._move(layer.displayed_coordinates)

        # if a shape is being moved, update the thumbnail
        if layer._is_moving:
            update_thumbnail = True
        yield

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
    corner = np.array(layer.displayed_coordinates)
    data = np.array([corner, corner + size])
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='line'
    )


def add_ellipse(layer, event):
    """Add an ellipse."""
    size = layer._vertex_size * layer.scale_factor / 4
    corner = np.array(layer.displayed_coordinates)
    data = np.array(
        [corner, corner + [size, 0], corner + size, corner + [0, size]]
    )
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='ellipse'
    )


def add_rectangle(layer, event):
    """Add an rectangle."""
    size = layer._vertex_size * layer.scale_factor / 4
    corner = np.array(layer.displayed_coordinates)
    data = np.array(
        [corner, corner + [size, 0], corner + size, corner + [0, size]]
    )
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='rectangle'
    )


def _add_line_rectangle_ellipse(layer, event, data, shape_type):
    """Helper function for adding a a line, rectangle or ellipse."""

    # on press
    # Start drawing rectangle / ellipse / line
    data_full = layer.expand_shape(data)
    layer.add(data_full, shape_type=shape_type)
    layer.selected_data = {layer.nshapes - 1}
    layer._value = (layer.nshapes - 1, 4)
    layer._moving_value = copy(layer._value)
    layer.refresh()
    yield

    # on move
    while event.type == 'mouse_move':
        # Drag any selected shapes
        layer._move(layer.displayed_coordinates)
        yield

    # on release
    layer._finish_drawing()


def add_path_polygon(layer, event):
    """Add a path or polygon."""
    coord = layer.displayed_coordinates

    # on press
    if layer._is_creating is False:
        # Start drawing a path
        data = np.array([coord, coord])
        data_full = layer.expand_shape(data)
        layer.add(data_full, shape_type='path')
        layer.selected_data = {layer.nshapes - 1}
        layer._value = (layer.nshapes - 1, 1)
        layer._moving_value = copy(layer._value)
        layer._is_creating = True
        layer._set_highlight()
    else:
        # Add to an existing path or polygon
        index = layer._moving_value[0]
        if layer._mode == Mode.ADD_POLYGON:
            new_type = Polygon
        else:
            new_type = None
        vertices = layer._data_view.displayed_vertices[
            layer._data_view.displayed_index == index
        ]
        vertices = np.concatenate((vertices, [coord]), axis=0)
        # Change the selected vertex
        layer._value = (layer._value[0], layer._value[1] + 1)
        layer._moving_value = copy(layer._value)
        data_full = layer.expand_shape(vertices)
        layer._data_view.edit(index, data_full, new_type=new_type)
        layer._selected_box = layer.interaction_box(layer.selected_data)


def add_path_polygon_creating(layer, event):
    """While a path or polygon move next vertex to be added."""
    if layer._is_creating:
        layer._move(layer.displayed_coordinates)


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
    ind, loc = point_to_lines(layer.displayed_coordinates, all_edges)
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
    vertices = layer._data_view.displayed_vertices[
        layer._data_view.displayed_index == index
    ]
    if not closed:
        if int(ind) == 1 and loc < 0:
            ind = 0
        elif int(ind) == len(vertices) - 1 and loc > 1:
            ind = ind + 1

    # Insert new vertex at appropriate place in vertices of target shape
    vertices = np.insert(vertices, ind, [layer.displayed_coordinates], axis=0)
    with layer.events.set_data.blocker():
        data_full = layer.expand_shape(vertices)
        layer._data_view.edit(index, data_full, new_type=new_type)
        layer._selected_box = layer.interaction_box(layer.selected_data)
    layer.refresh()


def vertex_remove(layer, event):
    """Remove a vertex from a selected shape.

    If a vertex is clicked on remove it from the shape it is in. If this cause
    the shape to shrink to a size that no longer is valid remove the whole
    shape.
    """
    shape_under_cursor, vertex_under_cursor = layer._value
    if vertex_under_cursor is None:
        # No vertex was clicked on so return
        return

    # Have clicked on a current vertex so remove
    shape_type = type(layer._data_view.shapes[shape_under_cursor])
    if shape_type == Ellipse:
        # Removing vertex from ellipse not implemented
        return
    vertices = layer._data_view.displayed_vertices[
        layer._data_view.displayed_index == shape_under_cursor
    ]
    if len(vertices) <= 2:
        # If only 2 vertices present, remove whole shape
        with layer.events.set_data.blocker():
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            layer._data_view.remove(shape_under_cursor)
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    elif shape_type == Polygon and len(vertices) == 3:
        # If only 3 vertices of a polygon present remove
        with layer.events.set_data.blocker():
            if shape_under_cursor in layer.selected_data:
                layer.selected_data.remove(shape_under_cursor)
            layer._data_view.remove(shape_under_cursor)
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    else:
        if shape_type == Rectangle:
            # Deleting vertex from a rectangle creates a polygon
            new_type = Polygon
        else:
            new_type = None
        # Remove clicked on vertex
        vertices = np.delete(vertices, vertex_under_cursor, axis=0)
        with layer.events.set_data.blocker():
            data_full = layer.expand_shape(vertices)
            layer._data_view.edit(
                shape_under_cursor, data_full, new_type=new_type
            )
            shapes = layer.selected_data
            layer._selected_box = layer.interaction_box(shapes)
    layer.refresh()

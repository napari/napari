import numpy as np
from copy import copy
from ._shapes_constants import Mode
from ._shapes_models import Rectangle, Ellipse, Line, Path, Polygon
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
    if layer._value[1] is None:
        if shift and layer._value[0] is not None:
            if layer._value[0] in layer.selected_data:
                layer.selected_data.remove(layer._value[0])
                shapes = layer.selected_data
                layer._selected_box = layer.interaction_box(shapes)
            else:
                layer.selected_data.append(layer._value[0])
                shapes = layer.selected_data
                layer._selected_box = layer.interaction_box(shapes)
        elif layer._value[0] is not None:
            if layer._value[0] not in layer.selected_data:
                layer.selected_data = {layer._value[0]}
        else:
            layer.selected_data = set()
    layer._set_highlight()
    yield

    # on move
    while event.type == 'mouse_move':
        # Drag any selected shapes
        coord = [layer.coordinates[i] for i in layer.dims.displayed]
        layer._move(coord)
        yield

    # on release
    shift = 'Shift' in event.modifiers
    if not layer._is_moving and not layer._is_selecting and not shift:
        if layer._value[0] is not None:
            layer.selected_data = {layer._value[0]}
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
    layer._update_thumbnail()


def add_line(layer, event):
    """Add a line."""
    coord = [layer.coordinates[i] for i in layer.dims.displayed]
    size = layer._vertex_size * layer.scale_factor / 4
    corner = np.array(coord)
    data = np.array([corner, corner + size])
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='line'
    )


def add_ellipse(layer, event):
    """Add an ellipse."""
    coord = [layer.coordinates[i] for i in layer.dims.displayed]
    size = layer._vertex_size * layer.scale_factor / 4
    corner = np.array(coord)
    data = np.array(
        [corner, corner + [size, 0], corner + size, corner + [0, size]]
    )
    yield from _add_line_rectangle_ellipse(
        layer, event, data=data, shape_type='ellipse'
    )


def add_rectangle(layer, event):
    """Add an rectangle."""
    coord = [layer.coordinates[i] for i in layer.dims.displayed]
    size = layer._vertex_size * layer.scale_factor / 4
    corner = np.array(coord)
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
        coord = [layer.coordinates[i] for i in layer.dims.displayed]
        layer._move(coord)
        yield

    # on release
    layer._finish_drawing()


def add_path_polygon(layer, event):
    """Add a path or polygon."""
    coord = [layer.coordinates[i] for i in layer.dims.displayed]

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
        coord = [layer.coordinates[i] for i in layer.dims.displayed]
        layer._move(coord)


def vertex_insert(layer, event):
    """Insert a vertex into a selected shape."""
    coord = [layer.coordinates[i] for i in layer.dims.displayed]
    if len(layer.selected_data) == 0:
        # If none selected return immediately
        return

    all_lines = np.empty((0, 2, 2))
    all_lines_shape = np.empty((0, 2), dtype=int)
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
            all_lines = np.append(all_lines, lines, axis=0)
            indices = np.array(
                [np.repeat(index, len(lines)), list(range(len(lines)))]
            ).T
            all_lines_shape = np.append(all_lines_shape, indices, axis=0)
    if len(all_lines) == 0:
        # No appropriate shapes found
        return
    ind, loc = point_to_lines(coord, all_lines)
    index = all_lines_shape[ind][0]
    ind = all_lines_shape[ind][1] + 1
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
    if closed is not True:
        if int(ind) == 1 and loc < 0:
            ind = 0
        elif int(ind) == len(vertices) - 1 and loc > 1:
            ind = ind + 1

    vertices = np.insert(vertices, ind, [coord], axis=0)
    with layer.events.set_data.blocker():
        data_full = layer.expand_shape(vertices)
        layer._data_view.edit(index, data_full, new_type=new_type)
        layer._selected_box = layer.interaction_box(layer.selected_data)
    layer.refresh()


def vertex_remove(layer, event):
    """Remove a vertex from a selected shape."""
    if layer._value[1] is not None:
        # have clicked on a current vertex so remove
        index = layer._value[0]
        shape_type = type(layer._data_view.shapes[index])
        if shape_type == Ellipse:
            # Removing vertex from ellipse not implemented
            return
        vertices = layer._data_view.displayed_vertices[
            layer._data_view.displayed_index == index
        ]
        if len(vertices) <= 2:
            # If only 2 vertices present, remove whole shape
            with layer.events.set_data.blocker():
                if index in layer.selected_data:
                    layer.selected_data.remove(index)
                layer._data_view.remove(index)
                shapes = layer.selected_data
                layer._selected_box = layer.interaction_box(shapes)
        elif shape_type == Polygon and len(vertices) == 3:
            # If only 3 vertices of a polygon present remove
            with layer.events.set_data.blocker():
                if index in layer.selected_data:
                    layer.selected_data.remove(index)
                layer._data_view.remove(index)
                shapes = layer.selected_data
                layer._selected_box = layer.interaction_box(shapes)
        else:
            if shape_type == Rectangle:
                # Deleting vertex from a rectangle creates a polygon
                new_type = Polygon
            else:
                new_type = None
            # Remove clicked on vertex
            vertices = np.delete(vertices, layer._value[1], axis=0)
            with layer.events.set_data.blocker():
                data_full = layer.expand_shape(vertices)
                layer._data_view.edit(index, data_full, new_type=new_type)
                shapes = layer.selected_data
                layer._selected_box = layer.interaction_box(shapes)
        layer.refresh()

import numpy as np

from ._shapes_constants import Box, Mode
from ._shapes_mouse_bindings import _move
from .shapes import Shapes


@Shapes.bind_key('Space')
def hold_to_pan_zoom(layer):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        prev_selected = layer.selected_data.copy()
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode
        layer.selected_data = prev_selected
        layer._set_highlight()


@Shapes.bind_key('Shift')
def hold_to_lock_aspect_ratio(layer):
    """Hold to lock aspect ratio when resizing a shape."""
    # on key press
    layer._fixed_aspect = True
    box = layer._selected_box
    if box is not None:
        size = box[Box.BOTTOM_RIGHT] - box[Box.TOP_LEFT]
        if not np.any(size == np.zeros(2)):
            layer._aspect_ratio = abs(size[1] / size[0])
        else:
            layer._aspect_ratio = 1
    else:
        layer._aspect_ratio = 1
    if layer._is_moving:
        _move(layer, layer._moving_coordinates)

    yield

    # on key release
    layer._fixed_aspect = False
    if layer._is_moving:
        _move(layer, layer._moving_coordinates)


@Shapes.bind_key('R')
def activate_add_rectangle_mode(layer):
    """Activate add rectangle tool."""
    layer.mode = Mode.ADD_RECTANGLE


@Shapes.bind_key('E')
def activate_add_ellipse_mode(layer):
    """Activate add ellipse tool."""
    layer.mode = Mode.ADD_ELLIPSE


@Shapes.bind_key('L')
def activate_add_line_mode(layer):
    """Activate add line tool."""
    layer.mode = Mode.ADD_LINE


@Shapes.bind_key('T')
def activate_add_path_mode(layer):
    """Activate add path tool."""
    layer.mode = Mode.ADD_PATH


@Shapes.bind_key('P')
def activate_add_polygon_mode(layer):
    """Activate add polygon tool."""
    layer.mode = Mode.ADD_POLYGON


@Shapes.bind_key('D')
def activate_direct_mode(layer):
    """Activate vertex selection tool."""
    layer.mode = Mode.DIRECT


@Shapes.bind_key('S')
def activate_select_mode(layer):
    """Activate shape selection tool."""
    layer.mode = Mode.SELECT


@Shapes.bind_key('Z')
def activate_pan_zoom_mode(layer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@Shapes.bind_key('I')
def activate_vertex_insert_mode(layer):
    """Activate vertex insertion tool."""
    layer.mode = Mode.VERTEX_INSERT


@Shapes.bind_key('X')
def activate_vertex_remove_mode(layer):
    """Activate vertex deletion tool."""
    layer.mode = Mode.VERTEX_REMOVE


@Shapes.bind_key('Control-C')
def copy(layer):
    """Copy any selected shapes."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._copy_data()


@Shapes.bind_key('Control-V')
def paste(layer):
    """Paste any copied shapes."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._paste_data()


@Shapes.bind_key('A')
def select_all(layer):
    """Select all shapes in the current view slice."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer.selected_data = set(np.nonzero(layer._data_view._displayed)[0])
        layer._set_highlight()


@Shapes.bind_key('Backspace')
def delete_selected(layer):
    """Delete any selected shapes."""
    layer.remove_selected()


@Shapes.bind_key('Escape')
def finish_drawing(layer):
    """Finish any drawing, for example when using the path or polygon tool."""
    layer._finish_drawing()

import numpy as np

from .shapes import Shapes
from ._constants import Mode, Box


@Shapes.bind_key('Space')
def hold_to_pan_zoom(layer):
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
        layer._move(layer.coordinates[-2:])

    yield

    # on key release
    layer._fixed_aspect = False
    if layer._is_moving:
        layer._move(layer.coordinates[-2:])


@Shapes.bind_key('R')
def activate_add_rectangle_mode(layer):
    layer.mode = Mode.ADD_RECTANGLE


@Shapes.bind_key('E')
def activate_add_ellipse_mode(layer):
    layer.mode = Mode.ADD_ELLIPSE


@Shapes.bind_key('L')
def activate_add_line_mode(layer):
    layer.mode = Mode.ADD_LINE


@Shapes.bind_key('T')
def activate_add_path_mode(layer):
    layer.mode = Mode.ADD_PATH


@Shapes.bind_key('P')
def activate_add_polygon_mode(layer):
    layer.mode = Mode.ADD_POLYGON


@Shapes.bind_key('D')
def activate_direct_mode(layer):
    layer.mode = Mode.DIRECT


@Shapes.bind_key('S')
def activate_select_mode(layer):
    layer.mode = Mode.SELECT


@Shapes.bind_key('Z')
def activate_pan_zoom_mode(layer):
    layer.mode = Mode.PAN_ZOOM


@Shapes.bind_key('I')
def activate_vertex_insert_mode(layer):
    layer.mode = Mode.VERTEX_INSERT


@Shapes.bind_key('X')
def activate_vertex_remove_mode(layer):
    layer.mode = Mode.VERTEX_REMOVE


@Shapes.bind_key('Control-C')
def copy(layer):
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._copy_data()


@Shapes.bind_key('Control-V')
def paste(layer):
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._paste_data()


@Shapes.bind_key('A')
def select_all(layer):
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer.selected_data = list(range(layer._nshapes_view))
        layer._set_highlight()


Shapes.bind_key('Backspace', Shapes.remove_selected)
Shapes.bind_key('Escape', Shapes._finish_drawing)

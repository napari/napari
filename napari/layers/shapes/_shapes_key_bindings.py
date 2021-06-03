import numpy as np

from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
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


def register_shapes_action(description, shortcuts=()):
    return register_layer_action(Shapes, description, shortcuts)


@register_shapes_action(trans._('Add rectangles'), "R")
def activate_add_rectangle_mode(layer):
    """Activate add rectangle tool."""
    layer.mode = Mode.ADD_RECTANGLE


@register_shapes_action(trans._('Add ellipses'), "E")
def activate_add_ellipse_mode(layer):
    """Activate add ellipse tool."""
    layer.mode = Mode.ADD_ELLIPSE


@register_shapes_action(trans._('Add lines'), "L")
def activate_add_line_mode(layer):
    """Activate add line tool."""
    layer.mode = Mode.ADD_LINE


@register_shapes_action(trans._('Add path'), "T")
def activate_add_path_mode(layer):
    """Activate add path tool."""
    layer.mode = Mode.ADD_PATH


@register_shapes_action(trans._('Add polygons'), "P")
def activate_add_polygon_mode(layer):
    """Activate add polygon tool."""
    layer.mode = Mode.ADD_POLYGON


@register_shapes_action(trans._('Select vertices'), "D")
def activate_direct_mode(layer):
    """Activate vertex selection tool."""
    layer.mode = Mode.DIRECT


@register_shapes_action(trans._('Select shapes'), "S")
def activate_select_mode(layer):
    """Activate shape selection tool."""
    layer.mode = Mode.SELECT


@register_shapes_action(trans._('Pan/Zoom'), "Z")
def activate_shape_pan_zoom_mode(layer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@register_shapes_action(trans._('Insert vertex'), "I")
def activate_vertex_insert_mode(layer):
    """Activate vertex insertion tool."""
    layer.mode = Mode.VERTEX_INSERT


@register_shapes_action(trans._('Remove vertex'), "X")
def activate_vertex_remove_mode(layer):
    """Activate vertex deletion tool."""
    layer.mode = Mode.VERTEX_REMOVE


@register_shapes_action(trans._('Copy any selected shapes'), "Control-C")
def copy_selected_shapes(layer):
    """Copy any selected shapes."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._copy_data()


@register_shapes_action(trans._('Paste any copied shapes'), "Control-V")
def paste_shape(layer):
    """Paste any copied shapes."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer._paste_data()


@register_shapes_action(
    trans._('Select all shapes in the current view slice'), "A"
)
def select_all_shapes(layer):
    """Select all shapes in the current view slice."""
    if layer._mode in (Mode.DIRECT, Mode.SELECT):
        layer.selected_data = set(np.nonzero(layer._data_view._displayed)[0])
        layer._set_highlight()


@register_shapes_action(
    trans._('Delete any selected shapes'), ("Backspace", "Delete")
)
def delete_selected_shapes(layer):
    """."""
    layer.remove_selected()


@register_shapes_action(trans._('Move to front'))
def move_shapes_selection_to_front(layer):
    layer.move_to_front()


@register_shapes_action(trans._('Move to back'))
def move_shapes_selection_to_back(layer):
    layer.move_to_back()


@register_shapes_action(
    trans._(
        'Finish any drawing, for example when using the path or polygon tool.'
    ),
    "Escape",
)
def finish_drawing_shape(layer):
    """Finish any drawing, for example when using the path or polygon tool."""
    layer._finish_drawing()

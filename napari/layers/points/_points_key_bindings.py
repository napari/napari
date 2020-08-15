from ._points_constants import Mode
from .points import Points


@Points.bind_key('Space')
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


@Points.bind_key('P')
def activate_add_mode(layer):
    """Activate add points tool."""
    layer.mode = Mode.ADD


@Points.bind_key('S')
def activate_select_mode(layer):
    """Activate select points tool."""
    layer.mode = Mode.SELECT


@Points.bind_key('Z')
def activate_pan_zoom_mode(layer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@Points.bind_key('Control-C')
def copy(layer):
    """Copy any selected points."""
    if layer._mode == Mode.SELECT:
        layer._copy_data()


@Points.bind_key('Control-V')
def paste(layer):
    """Paste any copied points."""
    if layer._mode == Mode.SELECT:
        layer._paste_data()


@Points.bind_key('A')
def select_all(layer):
    """Select all points in the current view slice."""
    if layer._mode == Mode.SELECT:
        layer.selected_data = set(layer._indices_view[: len(layer._view_data)])
        layer._set_highlight()


@Points.bind_key('Backspace')
@Points.bind_key('Delete')
def delete_selected(layer):
    """Delete all selected points."""
    if layer._mode in (Mode.SELECT, Mode.ADD):
        layer.remove_selected()

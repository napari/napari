from .annotations import Annotations
from ._constants import Mode


@Annotations.bind_key('Space')
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


@Annotations.bind_key('P')
def activate_add_mode(layer):
    """Activate add points tool."""
    layer.mode = Mode.ADD


@Annotations.bind_key('S')
def activate_select_mode(layer):
    """Activate select points tool."""
    layer.mode = Mode.SELECT


@Annotations.bind_key('Z')
def activate_pan_zoom_mode(layer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@Annotations.bind_key('Control-C')
def copy(layer):
    """Copy any selected points."""
    if layer._mode == Mode.SELECT:
        layer._copy_data()


@Annotations.bind_key('Control-V')
def paste(layer):
    """Paste any copied points."""
    if layer._mode == Mode.SELECT:
        layer._paste_data()


@Annotations.bind_key('A')
def select_all(layer):
    """Select all points in the current view slice."""
    if layer._mode == Mode.SELECT:
        layer.selected_data = layer._indices_view[: len(layer._data_view)]
        layer._set_highlight()


@Annotations.bind_key('Backspace')
def delete_selected(layer):
    """Delet all selected points."""
    if layer._mode == Mode.SELECT:
        layer.remove_selected()

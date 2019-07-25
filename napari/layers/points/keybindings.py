from .points import Points
from ._constants import Mode


@Points.bind_key('Space')
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


@Points.bind_key('P')
def activate_add_mode(layer):
    layer.mode = Mode.ADD


@Points.bind_key('S')
def activate_select_mode(layer):
    layer.mode = Mode.SELECT


@Points.bind_key('Z')
def activate_pan_zoom_mode(layer):
    layer.mode = Mode.PAN_ZOOM


@Points.bind_key('Control-C')
def copy(layer):
    if layer._mode == Mode.SELECT:
        layer._copy_data()


@Points.bind_key('Control-V')
def paste(layer):
    if layer._mode == Mode.SELECT:
        layer._paste_data()


@Points.bind_key('A')
def select_all(layer):
    if layer._mode == Mode.SELECT:
        layer.selected_data = layer._indices_view[: len(layer._data_view)]
        layer._set_highlight()


@Points.bind_key('Backspace')
def delete_selected(layer):
    if layer._mode == Mode.SELECT:
        layer.remove_selected()

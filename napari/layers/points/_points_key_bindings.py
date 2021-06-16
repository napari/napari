from __future__ import annotations

from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._points_constants import Mode
from .points import Points


def register_points_action(description, invert=None):
    return register_layer_action(Points, description, invert=invert)


@register_points_action(trans._('Add points'))
def activate_points_add_mode(layer):
    layer.mode = Mode.ADD


@register_points_action(trans._('Select points'))
def activate_points_select_mode(layer):
    layer.mode = Mode.SELECT


def _pan_zoom_invert(layer, state):
    """Hold to pan and zoom in the viewer."""
    prev_selected, prev_mode = state
    layer.mode = prev_mode
    layer.selected_data = prev_selected
    layer._set_highlight()


@register_points_action(trans._('Pan/zoom'), invert=_pan_zoom_invert)
def activate_points_pan_zoom_mode(layer):
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        prev_selected = layer.selected_data.copy()
        layer.mode = Mode.PAN_ZOOM
        return prev_selected, prev_mode


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


@register_points_action(
    trans._("Select all points in the current view slice."),
)
def select_all(layer):
    if layer._mode == Mode.SELECT:
        layer.selected_data = set(layer._indices_view[: len(layer._view_data)])
        layer._set_highlight()


@register_points_action(trans._('Delete selected points'))
def delete_selected_points(layer):
    """Delete all selected points."""
    if layer._mode in (Mode.SELECT, Mode.ADD):
        layer.remove_selected()

from __future__ import annotations

from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._points_constants import Mode
from .points import Points


def register_points_action(description):
    return register_layer_action(Points, description)


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


@register_points_action(trans._('Add points'))
def activate_points_add_mode(layer):
    layer.mode = Mode.ADD


@register_points_action(trans._('Select points'))
def activate_points_select_mode(layer):
    layer.mode = Mode.SELECT


@register_points_action(trans._('Pan/zoom'))
def activate_points_pan_zoom_mode(layer):
    layer.mode = Mode.PAN_ZOOM


@Points.bind_key('Control-C')
def copy(layer):
    """Copy any selected points."""
    layer._copy_data()


@Points.bind_key('Control-V')
def paste(layer):
    """Paste any copied points."""
    layer._paste_data()


@register_points_action(
    trans._("Select all points in the current view slice."),
)
def select_all(layer):
    layer.selected_data = set(layer._indices_view[: len(layer._view_data)])
    layer._set_highlight()


@register_points_action(trans._('Delete selected points'))
def delete_selected_points(layer):
    """Delete all selected points."""
    layer.remove_selected()

from __future__ import annotations

from app_model.types import KeyCode, KeyMod

from napari.utils.notifications import show_info

from ...layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from ...utils.translations import trans
from ._points_constants import Mode
from .points import Points


def register_points_action(description: str, repeatable: bool = False):
    return register_layer_action(Points, description, repeatable)


def register_points_mode_action(description):
    return register_layer_attr_action(Points, description, 'mode')


@Points.bind_key(KeyCode.Space)
def hold_to_pan_zoom(layer: Points):
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


@register_points_mode_action(trans._('Add points'))
def activate_points_add_mode(layer: Points):
    layer.mode = Mode.ADD


@register_points_mode_action(trans._('Select points'))
def activate_points_select_mode(layer: Points):
    layer.mode = Mode.SELECT


@register_points_mode_action(trans._('Pan/zoom'))
def activate_points_pan_zoom_mode(layer: Points):
    layer.mode = Mode.PAN_ZOOM


points_fun_to_mode = [
    (activate_points_add_mode, Mode.ADD),
    (activate_points_select_mode, Mode.SELECT),
    (activate_points_pan_zoom_mode, Mode.PAN_ZOOM),
]


@Points.bind_key(KeyMod.CtrlCmd | KeyCode.KeyC)
def copy(layer: Points):
    """Copy any selected points."""
    layer._copy_data()


@Points.bind_key(KeyMod.CtrlCmd | KeyCode.KeyV)
def paste(layer: Points):
    """Paste any copied points."""
    layer._paste_data()


@register_points_action(
    trans._("Select all points in the current view slice."),
)
def select_all_in_slice(layer: Points):
    new_selected = set(layer._indices_view[: len(layer._view_data)])

    # If all visible points are already selected, deselect the visible points
    if new_selected & layer.selected_data == new_selected:
        layer.selected_data = layer.selected_data - new_selected
        show_info(
            trans._(
                "Deselected all points in this slice, use Shift-A to deselect all points on the layer. ({n_total} selected)",
                n_total=len(layer.selected_data),
                deferred=True,
            )
        )

    # If not all visible points are already selected, additionally select the visible points
    else:
        layer.selected_data = layer.selected_data | new_selected
        show_info(
            trans._(
                "Selected {n_new} points in this slice, use Shift-A to select all points on the layer. ({n_total} selected)",
                n_new=len(new_selected),
                n_total=len(layer.selected_data),
                deferred=True,
            )
        )
    layer._set_highlight()


@register_points_action(
    trans._("Select all points in the layer."),
)
def select_all_data(layer: Points):

    # If all points are already selected, deselect all points
    if len(layer.selected_data) == len(layer.data):
        layer.selected_data = set()
        show_info(trans._("Cleared all selections.", deferred=True))

    # Select all points
    else:
        new_selected = set(range(layer.data.shape[0]))
        # Needed for the notification
        view_selected = set(layer._indices_view[: len(layer._view_data)])
        layer.selected_data = new_selected
        show_info(
            trans._(
                "Selected {n_new} points across all slices, including {n_invis} points not currently visible. ({n_total})",
                n_new=len(new_selected),
                n_invis=len(new_selected - view_selected),
                n_total=len(layer.selected_data),
                deferred=True,
            )
        )
    layer._set_highlight()


@register_points_action(trans._('Delete selected points'))
def delete_selected_points(layer: Points):
    """Delete all selected points."""
    layer.remove_selected()

from __future__ import annotations

from napari.layers.points._points_constants import Mode
from napari.layers.points.points import Points
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.utils.notifications import show_info
from napari.utils.translations import trans


def register_points_action(description: str, repeatable: bool = False):
    return register_layer_action(Points, description, repeatable)


def register_points_mode_action(description):
    return register_layer_attr_action(Points, description, 'mode')


@register_points_mode_action(trans._('Transform'))
def activate_points_transform_mode(layer):
    layer.mode = Mode.TRANSFORM


@register_points_mode_action(trans._('Pan/zoom'))
def activate_points_pan_zoom_mode(layer: Points):
    layer.mode = Mode.PAN_ZOOM


@register_points_mode_action(trans._('Add points'))
def activate_points_add_mode(layer: Points):
    layer.mode = Mode.ADD


@register_points_mode_action(trans._('Select points'))
def activate_points_select_mode(layer: Points):
    layer.mode = Mode.SELECT


points_fun_to_mode = [
    (activate_points_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_points_transform_mode, Mode.TRANSFORM),
    (activate_points_add_mode, Mode.ADD),
    (activate_points_select_mode, Mode.SELECT),
]


@Points.bind_key('Control-C')
def copy(layer: Points):
    """Copy any selected points."""
    layer._copy_data()


@Points.bind_key('Control-V')
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
    layer._set_highlight(force=True)


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
    layer._set_highlight(force=True)


@register_points_action(trans._('Delete selected points'))
def delete_selected_points(layer: Points):
    """Delete all selected points."""
    layer.remove_selected()

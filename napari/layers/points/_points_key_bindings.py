from __future__ import annotations

from napari.utils.notifications import show_info

from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._points_constants import Mode
from .points import Points


def register_points_action(description):
    return register_layer_action(Points, description)


@Points.bind_key('Space')
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


@register_points_action(trans._('Add points'))
def activate_points_add_mode(layer: Points):
    layer.mode = Mode.ADD


@register_points_action(trans._('Select points'))
def activate_points_select_mode(layer: Points):
    layer.mode = Mode.SELECT


@register_points_action(trans._('Pan/zoom'))
def activate_points_pan_zoom_mode(layer: Points):
    layer.mode = Mode.PAN_ZOOM


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
def select_all(layer: Points):
    new_selected = set(layer._indices_view[: len(layer._view_data)])

    # If all visible points are already selected, deselect the visible points
    if new_selected & layer.selected_data == new_selected:
        layer.selected_data = layer.selected_data - new_selected
        show_info(
            trans._(
                f"Deselected all points in this slice, use Shift-A to deselect all points on the layer. ({len(layer.selected_data)} selected)"
            )
        )

    # If not all visible points are already selected, additionally select the visible points
    else:
        layer.selected_data = layer.selected_data | new_selected
        show_info(
            trans._(
                f"Selected {len(new_selected)} points in this slice, use Shift-A to select all points on the layer. ({len(layer.selected_data)} selected)"
            )
        )
    layer._set_highlight()


@register_points_action(
    trans._("Select all points in the layer."),
)
def select_all_3d(layer: Points):
    new_selected = set(range(layer.data.shape[0]))
    # Needed for the notification
    view_selected = set(layer._indices_view[: len(layer._view_data)])

    # If all points are already selected, deselect all points
    if layer.selected_data == new_selected:
        layer.selected_data = set()
        show_info(
            trans._(
                f"Deselected all points across all slices, including {len(new_selected - view_selected)} points not currently visible. ({len(layer.selected_data)} selected)"
            )
        )

    # Select all points
    else:
        layer.selected_data = new_selected
        show_info(
            trans._(
                f"Selected {len(new_selected)} points across all slices, including {len(new_selected - view_selected)} points not currently visible. ({len(layer.selected_data)} selected)"
            )
        )
    layer._set_highlight()


@register_points_action(trans._('Delete selected points'))
def delete_selected_points(layer: Points):
    """Delete all selected points."""
    layer.remove_selected()

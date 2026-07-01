from __future__ import annotations

from typing import TYPE_CHECKING

from app_model.types import KeyBinding, KeyCode, KeyMod

from napari.layers.points._points_constants import Mode
from napari.layers.points.points import Points
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.settings import get_settings
from napari.utils.interactions import Shortcut
from napari.utils.notifications import show_info

if TYPE_CHECKING:
    from collections.abc import Callable


def shortcuts() -> dict[str, list[KeyBinding]]:
    return get_settings().shortcuts.shortcuts


def register_points_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Points, description, repeatable)


def register_points_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Points, description, 'mode')


@register_points_mode_action('Transform')
def activate_points_transform_mode(layer: Points) -> None:
    layer.mode = Mode.TRANSFORM


@register_points_mode_action('Move camera')
def activate_points_pan_zoom_mode(layer: Points) -> None:
    layer.mode = Mode.PAN_ZOOM


@register_points_mode_action('Add points')
def activate_points_add_mode(layer: Points) -> None:
    layer.mode = Mode.ADD


@register_points_mode_action('Select points')
def activate_points_select_mode(layer: Points) -> None:
    layer.mode = Mode.SELECT


points_fun_to_mode = [
    (activate_points_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_points_transform_mode, Mode.TRANSFORM),
    (activate_points_add_mode, Mode.ADD),
    (activate_points_select_mode, Mode.SELECT),
]


@Points.bind_key(KeyMod.CtrlCmd | KeyCode.KeyC, overwrite=True)
def copy(layer: Points) -> None:
    """Copy any selected points."""
    layer._copy_data()


@Points.bind_key(KeyMod.CtrlCmd | KeyCode.KeyV, overwrite=True)
def paste(layer: Points) -> None:
    """Paste any copied points."""
    layer._paste_data()


@register_points_action(
    'Select/Deselect all points in the current view slice',
)
def select_all_in_slice(layer: Points) -> None:
    """Select only the points in the current view slice, don't append."""
    new_selected = set(layer._indices_view[: len(layer._view_data)])

    # If all visible points are already selected, deselect the visible points
    if new_selected & layer.selected_data == new_selected:
        layer.selected_data = layer.selected_data - new_selected
        show_info(
            f'Deselected all points in this slice, use {Shortcut(shortcuts()["napari:select_all_data"][0])} to select/deselect all points on the layer. ({len(layer.selected_data)} selected)'
        )

    # If visible points are not already selected, select just the visible points
    else:
        layer.selected_data = new_selected
        show_info(
            f'Selected {len(new_selected)} points in this slice only, '
            f'use {Shortcut(shortcuts()["napari:select_append_all_in_slice"][0])} '
            f'to append to existing selection. ({len(layer.selected_data)} selected)'
        )


@register_points_action(
    'Select/Deselect all points in the current view slice',
)
def select_append_all_in_slice(layer: Points) -> None:
    """Select all points in the current view slice, appending to existing selection"""
    new_selected = set(layer._indices_view[: len(layer._view_data)])

    # If all visible points are already selected, deselect the visible points
    if new_selected & layer.selected_data == new_selected:
        layer.selected_data = layer.selected_data - new_selected
        show_info(
            f'Deselected all points in this slice, use {Shortcut(shortcuts()["napari:select_all_data"][0])} to select/deselect all points on the layer. ({len(layer.selected_data)} selected)'
        )

    # If not all visible points are already selected, additionally select the visible points
    else:
        layer.selected_data = layer.selected_data | new_selected
        show_info(
            f'Appended {len(new_selected)} points in this slice to the selection, use {Shortcut(shortcuts()["napari:select_all_data"][0])} to select all points on the layer. ({len(layer.selected_data)} selected)'
        )


@register_points_action(
    'Select/Deselect all points in the layer',
)
def select_all_data(layer: Points) -> None:
    # If all points are already selected, deselect all points
    if len(layer.selected_data) == len(layer.data):
        layer.selected_data = set()
        show_info('Cleared all selections.')

    # Select all points
    else:
        new_selected = set(range(layer.data.shape[0]))
        # Needed for the notification
        view_selected = set(layer._indices_view[: len(layer._view_data)])
        layer.selected_data = new_selected
        show_info(
            f'Selected {len(new_selected)} points across all slices, including {len(new_selected - view_selected)} points not currently visible. ({len(layer.selected_data)})'
        )


@register_points_action('Delete selected points')
def delete_selected_points(layer: Points) -> None:
    """Delete all selected points."""
    layer.remove_selected()

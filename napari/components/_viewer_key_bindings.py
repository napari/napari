from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from app_model.types import KeyCode, KeyMod

from napari.components.viewer_model import ViewerModel
from napari.utils.action_manager import action_manager
from napari.utils.notifications import show_info
from napari.utils.theme import available_themes, get_system_theme
from napari.utils.transforms import Affine
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.viewer import Viewer


def register_viewer_action(description, repeatable=False):
    """
    Convenient decorator to register an action with the current ViewerModel

    It will use the function name as the action name. We force the description
    to be given instead of function docstring for translation purpose.
    """

    def _inner(func):
        action_manager.register_action(
            name=f'napari:{func.__name__}',
            command=func,
            description=description,
            keymapprovider=ViewerModel,
            repeatable=repeatable,
        )
        return func

    return _inner


@ViewerModel.bind_key(KeyMod.Shift | KeyCode.UpArrow, overwrite=True)
def extend_selection_to_layer_above(viewer: Viewer):
    viewer.layers.select_next(shift=True)


@ViewerModel.bind_key(KeyMod.Shift | KeyCode.DownArrow, overwrite=True)
def extend_selection_to_layer_below(viewer: Viewer):
    viewer.layers.select_previous(shift=True)


@register_viewer_action(trans._('Reset scroll'))
def reset_scroll_progress(viewer: Viewer):
    # on key press
    viewer.dims._scroll_progress = 0
    yield

    # on key release
    viewer.dims._scroll_progress = 0


reset_scroll_progress.__doc__ = trans._('Reset dims scroll progress')


@register_viewer_action(trans._('Toggle 2D/3D view'))
def toggle_ndisplay(viewer: Viewer):
    if viewer.dims.ndisplay == 2:
        viewer.dims.ndisplay = 3
    else:
        viewer.dims.ndisplay = 2


# Making this an action makes vispy really unhappy during the tests
# on mac only with:
# ```
# RuntimeError: wrapped C/C++ object of type CanvasBackendDesktop has been deleted
# ```
@register_viewer_action(trans._('Toggle current viewer theme'))
def toggle_theme(viewer: ViewerModel):
    """Toggle theme for current viewer"""
    themes = available_themes()
    current_theme = viewer.theme
    # Check what the system theme is, to toggle properly
    if current_theme == 'system':
        current_theme = get_system_theme()
    idx = themes.index(current_theme)
    idx = (idx + 1) % len(themes)
    # Don't toggle to system, just among actual themes
    if themes[idx] == 'system':
        idx = (idx + 1) % len(themes)

    viewer.theme = themes[idx]


@register_viewer_action(trans._('Reset view to original state'))
def reset_view(viewer: Viewer):
    viewer.reset_view()


@register_viewer_action(trans._('Delete selected layers'))
def delete_selected_layers(viewer: Viewer):
    viewer.layers.remove_selected()


@register_viewer_action(
    trans._('Increment dimensions slider to the left'), repeatable=True
)
def increment_dims_left(viewer: Viewer):
    viewer.dims._increment_dims_left()


@register_viewer_action(
    trans._('Increment dimensions slider to the right'), repeatable=True
)
def increment_dims_right(viewer: Viewer):
    viewer.dims._increment_dims_right()


@register_viewer_action(trans._('Move focus of dimensions slider up'))
def focus_axes_up(viewer: Viewer):
    viewer.dims._focus_up()


@register_viewer_action(trans._('Move focus of dimensions slider down'))
def focus_axes_down(viewer: Viewer):
    viewer.dims._focus_down()


# Use non-breaking spaces and non-breaking hyphen for Preferences table
@register_viewer_action(
    trans._(
        'Change order of the visible axes, e.g.\u00a0[0,\u00a01,\u00a02]\u00a0\u2011>\u00a0[2,\u00a00,\u00a01]'
    ),
)
def roll_axes(viewer: Viewer):
    viewer.dims.roll()


# Use non-breaking spaces and non-breaking hyphen for Preferences table
@register_viewer_action(
    trans._(
        'Transpose order of the last two visible axes, e.g.\u00a0[0,\u00a01]\u00a0\u2011>\u00a0[1,\u00a00]'
    ),
)
def transpose_axes(viewer: Viewer):
    viewer.dims.transpose()


@register_viewer_action(trans._('Rotate layers 90 degrees counter-clockwise'))
def rotate_layers(viewer: Viewer):
    if viewer.dims.ndisplay == 3:
        show_info(trans._('Rotating layers only works in 2D'))
        return
    for layer in viewer.layers:
        if layer.ndim == 2:
            visible_dims = [0, 1]
        else:
            visible_dims = list(viewer.dims.displayed)

        initial_affine = layer.affine.set_slice(visible_dims)
        # want to rotate around a fixed refernce for all layers
        center = (
            np.asarray(viewer.dims.range)[:, 0][
                np.asarray(viewer.dims.displayed)
            ]
            + (
                np.asarray(viewer.dims.range)[:, 1][
                    np.asarray(viewer.dims.displayed)
                ]
                - np.asarray(viewer.dims.range)[:, 0][
                    np.asarray(viewer.dims.displayed)
                ]
            )
            / 2
        )
        new_affine = (
            Affine(translate=center)
            .compose(Affine(rotate=90))
            .compose(Affine(translate=-center))
            .compose(initial_affine)
        )
        layer.affine = layer.affine.replace_slice(visible_dims, new_affine)


@register_viewer_action(trans._('Toggle grid mode'))
def toggle_grid(viewer: Viewer):
    viewer.grid.enabled = not viewer.grid.enabled


@register_viewer_action(trans._('Toggle visibility of selected layers'))
def toggle_selected_visibility(viewer: Viewer):
    viewer.layers.toggle_selected_visibility()


@register_viewer_action(trans._('Toggle visibility of unselected layers'))
def toggle_unselected_visibility(viewer: Viewer):
    for layer in viewer.layers:
        if layer not in viewer.layers.selection:
            layer.visible = not layer.visible


@register_viewer_action(trans._('Select and show only layer above'))
def show_only_layer_above(viewer):
    viewer.layers.select_next()
    _show_only_selected_layer(viewer)


@register_viewer_action(trans._('Select and show only layer below'))
def show_only_layer_below(viewer):
    viewer.layers.select_previous()
    _show_only_selected_layer(viewer)


@register_viewer_action(
    trans._(
        'Show/Hide IPython console (only available when napari started as standalone application)'
    )
)
def toggle_console_visibility(viewer: Viewer):
    viewer.window._qt_viewer.toggle_console_visibility()


@register_viewer_action(trans._('Press and hold for pan/zoom mode'))
def hold_for_pan_zoom(viewer: ViewerModel):
    selected_layer = viewer.layers.selection.active
    if selected_layer is None:
        yield
        return
    previous_mode = selected_layer.mode
    # Each layer has its own Mode enum class with different values,
    # but they should all have a PAN_ZOOM value. At the time of writing
    # these enums do not share a base class or protocol, so ignore the
    # attribute check for now.
    pan_zoom = selected_layer._modeclass.PAN_ZOOM  # type: ignore[attr-defined]
    if previous_mode != pan_zoom:
        selected_layer.mode = pan_zoom
        yield

        selected_layer.mode = previous_mode


@register_viewer_action(trans._('Show all key bindings'))
def show_shortcuts(viewer: Viewer):
    pref_list = viewer.window._open_preferences_dialog()._list
    for i in range(pref_list.count()):
        if (item := pref_list.item(i)) and item.text() == 'Shortcuts':
            pref_list.setCurrentRow(i)


def _show_only_selected_layer(viewer):
    """Helper function to show only selected layer"""
    for layer in viewer.layers:
        if layer not in viewer.layers.selection:
            layer.visible = False
        else:
            layer.visible = True

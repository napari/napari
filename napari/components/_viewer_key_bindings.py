from __future__ import annotations

from typing import TYPE_CHECKING

from napari.utils.theme import available_themes, get_system_theme

if TYPE_CHECKING:
    from napari.viewer import Viewer


def reset_scroll_progress(viewer: Viewer):
    # on key press
    viewer.dims._scroll_progress = 0
    yield
    # on key release
    viewer.dims._scroll_progress = 0


# Making this an action makes vispy really unhappy during the tests
# on mac only with:
# ```
# RuntimeError: wrapped C/C++ object of type CanvasBackendDesktop has been deleted
# ```
def toggle_theme(viewer: Viewer):
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


def reset_view(viewer: Viewer):
    viewer.reset_view()


def delete_selected_layers(viewer: Viewer):
    viewer.layers.remove_selected()


def increment_dims_left(viewer: Viewer):
    viewer.dims._increment_dims_left()


def increment_dims_right(viewer: Viewer):
    viewer.dims._increment_dims_right()


def focus_axes_up(viewer: Viewer):
    viewer.dims._focus_up()


def focus_axes_down(viewer: Viewer):
    viewer.dims._focus_down()


def roll_axes(viewer: Viewer):
    viewer.dims._roll()


def transpose_axes(viewer: Viewer):
    viewer.dims.transpose()


def toggle_selected_layer_visibility(viewer: Viewer):
    viewer.layers.toggle_selected_visibility()


def toggle_console_visibility(viewer: Viewer):
    viewer.window._qt_viewer.toggle_console_visibility()


def hold_for_pan_zoom(viewer: Viewer):
    selected_layer = viewer.layers.selection.active
    if selected_layer is None:
        yield
        return
    previous_mode = selected_layer.mode
    if previous_mode != selected_layer._modeclass.PAN_ZOOM:
        selected_layer.mode = selected_layer._modeclass.PAN_ZOOM
        yield

        selected_layer.mode = previous_mode


def show_shortcuts(viewer: Viewer):
    viewer.window.file_menu._open_preferences()
    pref_list = viewer.window.file_menu._pref_dialog._list
    for i in range(pref_list.count()):
        if pref_list.item(i).text() == "Shortcuts":
            pref_list.setCurrentRow(i)


def new_labels(viewer: Viewer):
    viewer._new_labels()


def new_shapes(viewer: Viewer):
    viewer._new_shapes()


def new_points(viewer: Viewer):
    viewer._new_points()

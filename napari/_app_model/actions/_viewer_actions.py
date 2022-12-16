"""Actions related to the 'View' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_view.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from app_model.types import Action

from napari._app_model.constants import CommandId
from napari.settings import get_settings
from napari.utils.theme import available_themes

if TYPE_CHECKING:
    from napari.viewer import Viewer


VIEWER_ACTIONS: List[Action] = []


def _reset_scroll_progress(viewer: Viewer):
    # on key press
    viewer.dims._scroll_progress = 0
    # TODO: app-model doesn't handle generators
    # action-manager seems to use this for keydown/up
    # yield
    # on key release
    # viewer.dims._scroll_progress = 0


def _cycle_theme(viewer: Viewer):
    """Toggle theme for viewer"""
    settings = get_settings()
    themes = available_themes()
    current_theme = settings.appearance.theme
    idx = (themes.index(current_theme) + 1) % len(themes)
    settings.appearance.theme = themes[idx]


def _reset_view(viewer: Viewer):
    viewer.reset_view()


def _increment_dims_left(viewer: Viewer):
    viewer.dims._increment_dims_left()


def _increment_dims_right(viewer: Viewer):
    viewer.dims._increment_dims_right()


def _focus_axes_up(viewer: Viewer):
    viewer.dims._focus_up()


def _focus_axes_down(viewer: Viewer):
    viewer.dims._focus_down()


def _roll_axes(viewer: Viewer):
    viewer.dims._roll()


def _transpose_axes(viewer: Viewer):
    viewer.dims.transpose()


def _toggle_selected_layer_visibility(viewer: Viewer):
    viewer.layers.toggle_selected_visibility()


def _toggle_console_visibility(viewer: Viewer):
    viewer.window._qt_viewer.toggle_console_visibility()


# actions ported to app_model from components/_viewer_key_bindings
VIEWER_ACTIONS.extend(
    [
        Action(
            id=CommandId.VIEWER_RESET_SCROLL,
            title=CommandId.VIEWER_RESET_SCROLL.description,
            short_title=CommandId.VIEWER_RESET_SCROLL.title,
            callback=_reset_scroll_progress,
        ),
        Action(
            id=CommandId.VIEWER_CYCLE_THEME,
            title=CommandId.VIEWER_CYCLE_THEME.title,
            callback=_cycle_theme,
        ),
        Action(
            id=CommandId.VIEWER_RESET_VIEW,
            title=CommandId.VIEWER_RESET_VIEW.description,
            short_title=CommandId.VIEWER_RESET_VIEW.title,
            callback=_reset_view,
        ),
        Action(
            id=CommandId.VIEWER_INC_DIMS_LEFT,
            title=CommandId.VIEWER_INC_DIMS_LEFT.title,
            callback=_increment_dims_left,
        ),
        Action(
            id=CommandId.VIEWER_INC_DIMS_RIGHT,
            title=CommandId.VIEWER_INC_DIMS_RIGHT.title,
            callback=_increment_dims_right,
        ),
        Action(
            id=CommandId.VIEWER_FOCUS_AXES_UP,
            short_title=CommandId.VIEWER_FOCUS_AXES_UP.title,
            title=CommandId.VIEWER_FOCUS_AXES_UP.description,
            callback=_focus_axes_up,
        ),
        Action(
            id=CommandId.VIEWER_FOCUS_AXES_DOWN,
            short_title=CommandId.VIEWER_FOCUS_AXES_DOWN.title,
            title=CommandId.VIEWER_FOCUS_AXES_DOWN.description,
            callback=_focus_axes_down,
        ),
        Action(
            id=CommandId.VIEWER_ROLL_AXES,
            short_title=CommandId.VIEWER_ROLL_AXES.title,
            title=CommandId.VIEWER_ROLL_AXES.description,
            callback=_roll_axes,
        ),
        Action(
            id=CommandId.VIEWER_TRANSPOSE_AXES,
            short_title=CommandId.VIEWER_TRANSPOSE_AXES.title,
            title=CommandId.VIEWER_TRANSPOSE_AXES.description,
            callback=_transpose_axes,
        ),
        Action(
            id=CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY,
            title=CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY.title,
            callback=_toggle_selected_layer_visibility,
        ),
        Action(
            id=CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY,
            title=CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY.title,
            callback=_toggle_console_visibility,
        ),
    ],
)

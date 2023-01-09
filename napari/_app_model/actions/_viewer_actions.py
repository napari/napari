"""Actions related to the viewer.

The Actions in VIEWER_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from app_model.types import Action, ToggleRule

from napari._app_model.actions import GeneratorCallback
from napari._app_model.actions._toggle_action import ViewerToggleAction
from napari._app_model.constants import CommandId, MenuId
from napari.components import _viewer_key_bindings as _viewer_actions

if TYPE_CHECKING:
    from napari.viewer import Viewer


VIEWER_ACTIONS: List[Action] = []

# actions ported to app_model from components/_viewer_key_bindings
VIEWER_ACTIONS = [
    Action(
        id=CommandId.VIEWER_RESET_SCROLL,
        title=CommandId.VIEWER_RESET_SCROLL.description,
        short_title=CommandId.VIEWER_RESET_SCROLL.title,
        callback=GeneratorCallback(_viewer_actions.reset_scroll_progress),
    ),
    Action(
        id=CommandId.VIEWER_TOGGLE_THEME,
        title=CommandId.VIEWER_TOGGLE_THEME.title,
        callback=_viewer_actions.toggle_theme,
    ),
    Action(
        id=CommandId.VIEWER_RESET_VIEW,
        title=CommandId.VIEWER_RESET_VIEW.description,
        short_title=CommandId.VIEWER_RESET_VIEW.title,
        callback=_viewer_actions.reset_view,
    ),
    Action(
        id=CommandId.VIEWER_INC_DIMS_LEFT,
        title=CommandId.VIEWER_INC_DIMS_LEFT.title,
        callback=_viewer_actions.increment_dims_left,
    ),
    Action(
        id=CommandId.VIEWER_INC_DIMS_RIGHT,
        title=CommandId.VIEWER_INC_DIMS_RIGHT.title,
        callback=_viewer_actions.increment_dims_right,
    ),
    Action(
        id=CommandId.VIEWER_FOCUS_AXES_UP,
        short_title=CommandId.VIEWER_FOCUS_AXES_UP.title,
        title=CommandId.VIEWER_FOCUS_AXES_UP.description,
        callback=_viewer_actions.focus_axes_up,
    ),
    Action(
        id=CommandId.VIEWER_FOCUS_AXES_DOWN,
        short_title=CommandId.VIEWER_FOCUS_AXES_DOWN.title,
        title=CommandId.VIEWER_FOCUS_AXES_DOWN.description,
        callback=_viewer_actions.focus_axes_down,
    ),
    Action(
        id=CommandId.VIEWER_ROLL_AXES,
        short_title=CommandId.VIEWER_ROLL_AXES.title,
        title=CommandId.VIEWER_ROLL_AXES.description,
        callback=_viewer_actions.roll_axes,
    ),
    Action(
        id=CommandId.VIEWER_TRANSPOSE_AXES,
        short_title=CommandId.VIEWER_TRANSPOSE_AXES.title,
        title=CommandId.VIEWER_TRANSPOSE_AXES.description,
        callback=_viewer_actions.transpose_axes,
    ),
    Action(
        id=CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY,
        title=CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY.title,
        callback=_viewer_actions.toggle_selected_layer_visibility,
    ),
    Action(
        id=CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY,
        title=CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY.title,
        callback=_viewer_actions.toggle_console_visibility,
    ),
]


def _ndisplay_toggle(viewer: Viewer):
    viewer.dims.ndisplay = 2 + (viewer.dims.ndisplay == 2)


def _get_current_ndisplay_is_3D(viewer: Viewer):
    return viewer.dims.ndisplay == 3


# these are separate because they include menu entries
VIEWER_ACTIONS.extend(
    [
        Action(
            id=CommandId.TOGGLE_VIEWER_NDISPLAY,
            title=CommandId.TOGGLE_VIEWER_NDISPLAY.title,
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 0}
            ],
            callback=_ndisplay_toggle,
            toggled=ToggleRule(get_current=_get_current_ndisplay_is_3D),
        ),
        ViewerToggleAction(
            id=CommandId.VIEWER_TOGGLE_GRID,
            title=CommandId.VIEWER_TOGGLE_GRID.title,
            viewer_attribute="grid",
            sub_attribute="enabled",
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 0},
            ],
        ),
    ],
)

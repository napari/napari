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
from napari._app_model.constants import DEFAULT_SHORTCUTS, CommandId, MenuId
from napari.components import _viewer_key_bindings as _viewer_actions

if TYPE_CHECKING:
    from napari.viewer import Viewer


# actions ported to app_model from components/_viewer_key_bindings
VIEWER_ACTIONS: List[Action] = [
    Action(
        id=CommandId.VIEWER_HOLD_FOR_PAN_ZOOM,
        title=CommandId.VIEWER_HOLD_FOR_PAN_ZOOM.description,
        short_title=CommandId.VIEWER_HOLD_FOR_PAN_ZOOM.title,
        callback=GeneratorCallback(_viewer_actions.hold_for_pan_zoom),
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_HOLD_FOR_PAN_ZOOM],
    ),
    Action(
        id=CommandId.VIEWER_DELETE_SELECTED,
        title=CommandId.VIEWER_DELETE_SELECTED.description,
        short_title=CommandId.VIEWER_DELETE_SELECTED.title,
        callback=_viewer_actions.delete_selected_layers,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_DELETE_SELECTED],
    ),
    Action(
        id=CommandId.VIEWER_RESET_SCROLL,
        title=CommandId.VIEWER_RESET_SCROLL.description,
        short_title=CommandId.VIEWER_RESET_SCROLL.title,
        callback=GeneratorCallback(_viewer_actions.reset_scroll_progress),
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_RESET_SCROLL],
    ),
    Action(
        id=CommandId.VIEWER_TOGGLE_THEME,
        title=CommandId.VIEWER_TOGGLE_THEME.title,
        callback=_viewer_actions.toggle_theme,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_TOGGLE_THEME],
    ),
    Action(
        id=CommandId.VIEWER_RESET_VIEW,
        title=CommandId.VIEWER_RESET_VIEW.description,
        short_title=CommandId.VIEWER_RESET_VIEW.title,
        callback=_viewer_actions.reset_view,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_RESET_VIEW],
    ),
    Action(
        id=CommandId.VIEWER_INC_DIMS_LEFT,
        title=CommandId.VIEWER_INC_DIMS_LEFT.title,
        callback=_viewer_actions.increment_dims_left,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_INC_DIMS_LEFT],
    ),
    Action(
        id=CommandId.VIEWER_INC_DIMS_RIGHT,
        title=CommandId.VIEWER_INC_DIMS_RIGHT.title,
        callback=_viewer_actions.increment_dims_right,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_INC_DIMS_RIGHT],
    ),
    Action(
        id=CommandId.VIEWER_FOCUS_AXES_UP,
        short_title=CommandId.VIEWER_FOCUS_AXES_UP.title,
        title=CommandId.VIEWER_FOCUS_AXES_UP.description,
        callback=_viewer_actions.focus_axes_up,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_FOCUS_AXES_UP],
    ),
    Action(
        id=CommandId.VIEWER_FOCUS_AXES_DOWN,
        short_title=CommandId.VIEWER_FOCUS_AXES_DOWN.title,
        title=CommandId.VIEWER_FOCUS_AXES_DOWN.description,
        callback=_viewer_actions.focus_axes_down,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_FOCUS_AXES_DOWN],
    ),
    Action(
        id=CommandId.VIEWER_ROLL_AXES,
        short_title=CommandId.VIEWER_ROLL_AXES.title,
        title=CommandId.VIEWER_ROLL_AXES.description,
        callback=_viewer_actions.roll_axes,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_ROLL_AXES],
    ),
    Action(
        id=CommandId.VIEWER_TRANSPOSE_AXES,
        short_title=CommandId.VIEWER_TRANSPOSE_AXES.title,
        title=CommandId.VIEWER_TRANSPOSE_AXES.description,
        callback=_viewer_actions.transpose_axes,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_TRANSPOSE_AXES],
    ),
    Action(
        id=CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY,
        title=CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY.title,
        callback=_viewer_actions.toggle_selected_layer_visibility,
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.VIEWER_TOGGLE_SELECTED_LAYER_VISIBILITY
        ],
    ),
    Action(
        id=CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY,
        title=CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY.title,
        callback=_viewer_actions.toggle_console_visibility,
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.VIEWER_TOGGLE_CONSOLE_VISIBILITY
        ],
    ),
    Action(
        id=CommandId.NAPARI_SHOW_SHORTCUTS,
        title=CommandId.NAPARI_SHOW_SHORTCUTS.title,
        callback=_viewer_actions.show_shortcuts,
        keybindings=DEFAULT_SHORTCUTS[CommandId.NAPARI_SHOW_SHORTCUTS],
    ),
    Action(
        id=CommandId.VIEWER_NEW_LABELS,
        title=CommandId.VIEWER_NEW_LABELS.title,
        callback=_viewer_actions.new_labels,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_NEW_LABELS],
    ),
    Action(
        id=CommandId.VIEWER_NEW_SHAPES,
        title=CommandId.VIEWER_NEW_SHAPES.title,
        callback=_viewer_actions.new_shapes,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_NEW_SHAPES],
    ),
    Action(
        id=CommandId.VIEWER_NEW_POINTS,
        title=CommandId.VIEWER_NEW_POINTS.title,
        callback=_viewer_actions.new_points,
        keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_NEW_POINTS],
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
            keybindings=DEFAULT_SHORTCUTS[CommandId.TOGGLE_VIEWER_NDISPLAY],
        ),
        ViewerToggleAction(
            id=CommandId.VIEWER_TOGGLE_GRID,
            title=CommandId.VIEWER_TOGGLE_GRID.title,
            viewer_attribute="grid",
            sub_attribute="enabled",
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 0},
            ],
            keybindings=DEFAULT_SHORTCUTS[CommandId.VIEWER_TOGGLE_GRID],
        ),
    ],
)

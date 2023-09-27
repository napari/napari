"""This module defines actions (functions) that operate on Points layers.

The Actions in POINTS_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from typing import List

from app_model.types import Action

from napari._app_model.actions import AttrRestoreCallback
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.constants import DEFAULT_SHORTCUTS, CommandId
from napari.layers.points import _points_key_bindings as _points_actions

enablement = LLSCK.active_layer_type == 'points'

# actions ported to app_model from components/_viewer_key_bindings
POINTS_ACTIONS: List[Action] = [
    Action(
        id=CommandId.POINTS_ACTIVATE_ADD_MODE,
        title=CommandId.POINTS_ACTIVATE_ADD_MODE.description,
        short_title=CommandId.POINTS_ACTIVATE_ADD_MODE.command_title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_add_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_ACTIVATE_ADD_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_ACTIVATE_SELECT_MODE,
        title=CommandId.POINTS_ACTIVATE_SELECT_MODE.description,
        short_title=CommandId.POINTS_ACTIVATE_SELECT_MODE.command_title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_select_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_ACTIVATE_SELECT_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE.description,
        short_title=CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE.command_title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_pan_zoom_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_ACTIVATE_TRANSFORM_MODE,
        title=CommandId.POINTS_ACTIVATE_TRANSFORM_MODE.description,
        short_title=CommandId.POINTS_ACTIVATE_TRANSFORM_MODE.command_title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_transform_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.POINTS_ACTIVATE_TRANSFORM_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_COPY,
        title=CommandId.POINTS_COPY.description,
        short_title=CommandId.POINTS_COPY.command_title,
        callback=_points_actions.copy,
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_COPY],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_PASTE,
        title=CommandId.POINTS_PASTE.description,
        short_title=CommandId.POINTS_PASTE.command_title,
        callback=_points_actions.paste,
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_PASTE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_SELECT_ALL_IN_SLICE,
        title=CommandId.POINTS_SELECT_ALL_IN_SLICE.description,
        short_title=CommandId.POINTS_SELECT_ALL_IN_SLICE.command_title,
        callback=_points_actions.select_all_in_slice,
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_SELECT_ALL_IN_SLICE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_SELECT_ALL_DATA,
        title=CommandId.POINTS_SELECT_ALL_DATA.description,
        short_title=CommandId.POINTS_SELECT_ALL_DATA.command_title,
        callback=_points_actions.select_all_data,
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_SELECT_ALL_DATA],
        enablement=enablement,
    ),
    Action(
        id=CommandId.POINTS_DELETE_SELECTED,
        title=CommandId.POINTS_DELETE_SELECTED.description,
        short_title=CommandId.POINTS_DELETE_SELECTED.command_title,
        callback=_points_actions.delete_selected_points,
        keybindings=DEFAULT_SHORTCUTS[CommandId.POINTS_DELETE_SELECTED],
        enablement=enablement,
    ),
]

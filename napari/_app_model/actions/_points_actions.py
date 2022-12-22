"""This module defines actions (functions) that operate on Points layers.

The Actions in POINTS_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action

from napari._app_model.actions import AttrRestoreCallback, GeneratorCallback
from napari._app_model.constants import CommandId
from napari.layers.points import _points_key_bindings as _points_actions

# actions ported to app_model from components/_viewer_key_bindings
POINTS_ACTIONS = [
    Action(
        id=CommandId.POINTS_HOLD_TO_PAN_ZOOM,
        title=CommandId.POINTS_HOLD_TO_PAN_ZOOM.description,
        short_title=CommandId.POINTS_HOLD_TO_PAN_ZOOM.title,
        callback=GeneratorCallback(_points_actions.hold_to_pan_zoom),
    ),
    Action(
        id=CommandId.POINTS_ACTIVATE_ADD_MODE,
        title=CommandId.POINTS_ACTIVATE_ADD_MODE.title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_add_mode, "mode"
        ),
    ),
    Action(
        id=CommandId.POINTS_ACTIVATE_SELECT_MODE,
        title=CommandId.POINTS_ACTIVATE_SELECT_MODE.title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_select_mode, "mode"
        ),
    ),
    Action(
        id=CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.POINTS_ACTIVATE_PAN_ZOOM_MODE.title,
        callback=AttrRestoreCallback(
            _points_actions.activate_points_pan_zoom_mode, "mode"
        ),
    ),
    Action(
        id=CommandId.POINTS_COPY,
        title=CommandId.POINTS_COPY.description,
        short_title=CommandId.POINTS_COPY.title,
        callback=_points_actions.copy,
    ),
    Action(
        id=CommandId.POINTS_PASTE,
        title=CommandId.POINTS_PASTE.description,
        short_title=CommandId.POINTS_PASTE.title,
        callback=_points_actions.paste,
    ),
    Action(
        id=CommandId.POINTS_SELECT_ALL_IN_SLICE,
        title=CommandId.POINTS_SELECT_ALL_IN_SLICE.description,
        short_title=CommandId.POINTS_SELECT_ALL_IN_SLICE.title,
        callback=_points_actions.select_all_in_slice,
    ),
    Action(
        id=CommandId.POINTS_SELECT_ALL_DATA,
        title=CommandId.POINTS_SELECT_ALL_DATA.description,
        short_title=CommandId.POINTS_SELECT_ALL_DATA.title,
        callback=_points_actions.select_all_data,
    ),
    Action(
        id=CommandId.POINTS_DELETE_SELECTED,
        title=CommandId.POINTS_DELETE_SELECTED.description,
        short_title=CommandId.POINTS_DELETE_SELECTED.title,
        callback=_points_actions.delete_selected_points,
    ),
]

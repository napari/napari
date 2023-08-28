"""This module defines actions (functions) that operate on Tracks layers.

The Actions in TRACKS_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action

from napari._app_model.constants import CommandId
from napari.layers.tracks import _tracks_key_bindings as _tracks_actions

# actions ported to app_model from layers/tracks/_tracks_key_bindings
TRACKS_ACTIONS = [
    Action(
        id=CommandId.TRACKS_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.TRACKS_ACTIVATE_PAN_ZOOM_MODE.description,
        short_title=CommandId.TRACKS_ACTIVATE_PAN_ZOOM_MODE.title,
        callback=_tracks_actions.activate_tracks_pan_zoom_mode,
    ),
    Action(
        id=CommandId.TRACKS_ACTIVATE_TRANSFORM_MODE,
        title=CommandId.TRACKS_ACTIVATE_TRANSFORM_MODE.description,
        short_title=CommandId.TRACKS_ACTIVATE_TRANSFORM_MODE.title,
        callback=_tracks_actions.activate_tracks_transform_mode,
    ),
]

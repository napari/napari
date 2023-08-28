"""This module defines actions (functions) that operate on Surface layers.

The Actions in SURFACE_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action

from napari._app_model.constants import CommandId
from napari.layers.surface import _surface_key_bindings as _surface_actions

# actions ported to app_model from layers/surface/_surface_key_bindings
SURFACE_ACTIONS = [
    Action(
        id=CommandId.SURFACE_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.SURFACE_ACTIVATE_PAN_ZOOM_MODE.description,
        short_title=CommandId.SURFACE_ACTIVATE_PAN_ZOOM_MODE.title,
        callback=_surface_actions.activate_surface_pan_zoom_mode,
    ),
    Action(
        id=CommandId.SURFACE_ACTIVATE_TRANSFORM_MODE,
        title=CommandId.SURFACE_ACTIVATE_TRANSFORM_MODE.description,
        short_title=CommandId.SURFACE_ACTIVATE_TRANSFORM_MODE.title,
        callback=_surface_actions.activate_surface_transform_mode,
    ),
]

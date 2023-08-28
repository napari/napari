"""This module defines actions (functions) that operate on Vectors layers.

The Actions in VECTORS_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action

from napari._app_model.constants import CommandId
from napari.layers.vectors import _vectors_key_bindings as _vectors_actions

# actions ported to app_model from layers/vectors/_vectors_key_bindings
VECTORS_ACTIONS = [
    Action(
        id=CommandId.VECTORS_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.VECTORS_ACTIVATE_PAN_ZOOM_MODE.description,
        short_title=CommandId.VECTORS_ACTIVATE_PAN_ZOOM_MODE.title,
        callback=_vectors_actions.activate_vectors_pan_zoom_mode,
    ),
    Action(
        id=CommandId.VECTORS_ACTIVATE_TRANSFORM_MODE,
        title=CommandId.VECTORS_ACTIVATE_TRANSFORM_MODE.description,
        short_title=CommandId.VECTORS_ACTIVATE_TRANSFORM_MODE.title,
        callback=_vectors_actions.activate_vectors_transform_mode,
    ),
]

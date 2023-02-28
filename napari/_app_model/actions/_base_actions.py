"""This module defines actions (functions) that operate on all layers.

The Actions in LAYER_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action

from napari._app_model.actions import GeneratorCallback
from napari._app_model.constants import CommandId
from napari.layers.base import _base_key_bindings as _base_actions

# actions ported to app_model from layers/base/_base_key_bindings
BASE_LAYER_ACTIONS = [
    Action(
        id=CommandId.BASE_HOLD_TO_PAN_ZOOM,
        title=CommandId.BASE_HOLD_TO_PAN_ZOOM.description,
        short_title=CommandId.BASE_HOLD_TO_PAN_ZOOM.title,
        callback=GeneratorCallback(_base_actions.hold_to_pan_zoom),
    ),
]

"""This module defines actions (functions) that operate on Image layers.

The Actions in IMAGE_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action

from napari._app_model.actions import GeneratorCallback
from napari._app_model.constants import CommandId
from napari.layers.image import _image_key_bindings as _image_actions

# actions ported to app_model from components/_viewer_key_bindings
IMAGE_ACTIONS = [
    Action(
        id=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Z,
        title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Z.description,
        short_title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Z.title,
        callback=_image_actions.orient_plane_normal_along_z,
    ),
    Action(
        id=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Y,
        title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Y.description,
        short_title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_Y.title,
        callback=_image_actions.orient_plane_normal_along_y,
    ),
    Action(
        id=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_X,
        title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_X.description,
        short_title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_X.title,
        callback=_image_actions.orient_plane_normal_along_x,
    ),
    Action(
        id=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION,
        title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION.description,
        short_title=CommandId.IMAGE_ORIENT_PLANE_NORMAL_ALONG_VIEW_DIRECTION.title,
        callback=_image_actions.orient_plane_normal_along_view_direction,
    ),
    Action(
        id=CommandId.IMAGE_HOLD_TO_PAN_ZOOM,
        title=CommandId.IMAGE_HOLD_TO_PAN_ZOOM.description,
        short_title=CommandId.IMAGE_HOLD_TO_PAN_ZOOM.title,
        callback=GeneratorCallback(_image_actions.hold_to_pan_zoom),
    ),
    Action(
        id=CommandId.IMAGE_ACTIVATE_TRANSFORM_MODE,
        title=CommandId.IMAGE_ACTIVATE_TRANSFORM_MODE.description,
        short_title=CommandId.IMAGE_ACTIVATE_TRANSFORM_MODE.title,
        callback=_image_actions.activate_image_transform_mode,
    ),
    Action(
        id=CommandId.IMAGE_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.IMAGE_ACTIVATE_PAN_ZOOM_MODE.description,
        short_title=CommandId.IMAGE_ACTIVATE_PAN_ZOOM_MODE.title,
        callback=_image_actions.activate_image_pan_zoom_mode,
    ),
]

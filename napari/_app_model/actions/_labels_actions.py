"""This module defines actions (functions) that operate on Labels layers.

The Actions in LABELS_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from app_model.types import Action, ToggleRule

from napari._app_model.actions import GeneratorCallback, RepeatableAction
from napari._app_model.constants import CommandId
from napari.layers.labels import _labels_key_bindings as _labels_actions

# actions ported to app_model from components/_viewer_key_bindings
LABELS_ACTIONS = [
    Action(
        id=CommandId.LABELS_HOLD_TO_PAN_ZOOM,
        title=CommandId.LABELS_HOLD_TO_PAN_ZOOM.description,
        short_title=CommandId.LABELS_HOLD_TO_PAN_ZOOM.title,
        callback=GeneratorCallback(_labels_actions.hold_to_pan_zoom),
    ),
    # TODO: register_labels_mode_action (register_layer_attr_action)
    # CommandId.LABELS_ACTIVATE_PAINT_MODE: _i(trans._('Paint'), trans._('activate the paint brush'),),
    # CommandId.LABELS_ACTIVATE_FILL_MODE: _i(trans._('Fill'), trans._('activate the fill bucket'),),
    # CommandId.LABELS_ACTIVATE_PAN_ZOOM_MODE: _i(trans._('Pan/zoom'), trans._('activate pan/zoom mode'),),
    # CommandId.LABELS_ACTIVATE_PICKER_MODE: _i(trans._('Pick mode'),),
    # CommandId.LABELS_ACTIVATE_ERASE_MODE: _i(trans._('Erase'), trans._('activate the label eraser'),),
    Action(
        id=CommandId.LABELS_NEW_LABEL,
        title=CommandId.LABELS_NEW_LABEL.description,
        short_title=CommandId.LABELS_NEW_LABEL.title,
        callback=_labels_actions.new_label,
    ),
    Action(
        id=CommandId.LABELS_DECREMENT_ID,
        title=CommandId.LABELS_DECREMENT_ID.description,
        short_title=CommandId.LABELS_DECREMENT_ID.title,
        callback=_labels_actions.decrease_label_id,
    ),
    Action(
        id=CommandId.LABELS_INCREMENT_ID,
        title=CommandId.LABELS_INCREMENT_ID.description,
        short_title=CommandId.LABELS_INCREMENT_ID.title,
        callback=_labels_actions.increase_label_id,
    ),
    RepeatableAction(
        id=CommandId.LABELS_DECREASE_BRUSH_SIZE,
        title=CommandId.LABELS_DECREASE_BRUSH_SIZE.title,
        callback=_labels_actions.decrease_brush_size,
    ),
    RepeatableAction(
        id=CommandId.LABELS_INCREASE_BRUSH_SIZE,
        title=CommandId.LABELS_INCREASE_BRUSH_SIZE.title,
        callback=_labels_actions.increase_brush_size,
    ),
    Action(
        id=CommandId.LABELS_TOGGLE_PRESERVE_LABELS,
        title=CommandId.LABELS_TOGGLE_PRESERVE_LABELS.title,
        callback=_labels_actions.toggle_preserve_labels,
        toggled=ToggleRule(
            get_current=_labels_actions._get_preserve_labels_toggled
        ),
    ),
    Action(
        id=CommandId.LABELS_UNDO,
        title=CommandId.LABELS_UNDO.description,
        short_title=CommandId.LABELS_UNDO.title,
        callback=_labels_actions.undo,
    ),
    Action(
        id=CommandId.LABELS_REDO,
        title=CommandId.LABELS_REDO.description,
        short_title=CommandId.LABELS_REDO.title,
        callback=_labels_actions.redo,
    ),
]

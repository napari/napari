"""This module defines actions (functions) that operate on Shapes layers.

The Actions in SHAPES_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from typing import List

from app_model.types import Action

from napari._app_model.actions import AttrRestoreCallback, GeneratorCallback
from napari._app_model.constants import DEFAULT_SHORTCUTS, CommandId
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.layers.shapes import _shapes_key_bindings as _shapes_actions

enablement = LLSCK.active_layer_type == 'shapes'

# actions ported to app_model from layers/shapes/_shapes_key_bindings
SHAPES_ACTIONS: List[Action] = [
    Action(
        id=CommandId.SHAPES_HOLD_TO_LOCK_ASPECT_RATIO,
        title=CommandId.SHAPES_HOLD_TO_LOCK_ASPECT_RATIO.description,
        short_title=CommandId.SHAPES_HOLD_TO_LOCK_ASPECT_RATIO.title,
        callback=GeneratorCallback(_shapes_actions.hold_to_lock_aspect_ratio),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_HOLD_TO_LOCK_ASPECT_RATIO
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_ADD_RECTANGLE_MODE,
        title=CommandId.SHAPES_ACTIVATE_ADD_RECTANGLE_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_ADD_RECTANGLE_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_add_rectangle_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_ADD_RECTANGLE_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_ADD_ELLIPSE_MODE,
        title=CommandId.SHAPES_ACTIVATE_ADD_ELLIPSE_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_ADD_ELLIPSE_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_add_ellipse_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_ADD_ELLIPSE_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_ADD_LINE_MODE,
        title=CommandId.SHAPES_ACTIVATE_ADD_LINE_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_ADD_LINE_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_add_line_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_ACTIVATE_ADD_LINE_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_ADD_PATH_MODE,
        title=CommandId.SHAPES_ACTIVATE_ADD_PATH_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_ADD_PATH_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_add_path_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_ACTIVATE_ADD_PATH_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_ADD_POLYGON_MODE,
        title=CommandId.SHAPES_ACTIVATE_ADD_POLYGON_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_ADD_POLYGON_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_add_polygon_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_ADD_POLYGON_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE,
        title=CommandId.SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_add_polygon_lasso_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_ADD_POLYGON_LASSO_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_DIRECT_MODE,
        title=CommandId.SHAPES_ACTIVATE_DIRECT_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_DIRECT_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_direct_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_ACTIVATE_DIRECT_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_SELECT_MODE,
        title=CommandId.SHAPES_ACTIVATE_SELECT_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_SELECT_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_select_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_ACTIVATE_SELECT_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_PAN_ZOOM_MODE,
        title=CommandId.SHAPES_ACTIVATE_PAN_ZOOM_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_PAN_ZOOM_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_shapes_pan_zoom_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_ACTIVATE_PAN_ZOOM_MODE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_TRANSFORM_MODE,
        title=CommandId.SHAPES_ACTIVATE_TRANSFORM_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_TRANSFORM_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_shapes_transform_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_TRANSFORM_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_VERTEX_INSERT_MODE,
        title=CommandId.SHAPES_ACTIVATE_VERTEX_INSERT_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_VERTEX_INSERT_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_vertex_insert_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_VERTEX_INSERT_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_ACTIVATE_VERTEX_REMOVE_MODE,
        title=CommandId.SHAPES_ACTIVATE_VERTEX_REMOVE_MODE.description,
        short_title=CommandId.SHAPES_ACTIVATE_VERTEX_REMOVE_MODE.title,
        callback=AttrRestoreCallback(
            _shapes_actions.activate_vertex_remove_mode, "mode"
        ),
        keybindings=DEFAULT_SHORTCUTS[
            CommandId.SHAPES_ACTIVATE_VERTEX_REMOVE_MODE
        ],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_COPY,
        title=CommandId.SHAPES_COPY.description,
        short_title=CommandId.SHAPES_COPY.title,
        callback=_shapes_actions.copy_selected_shapes,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_COPY],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_PASTE,
        title=CommandId.SHAPES_PASTE.description,
        short_title=CommandId.SHAPES_PASTE.title,
        callback=_shapes_actions.paste_shape,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_PASTE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_SELECT_ALL,
        title=CommandId.SHAPES_SELECT_ALL.description,
        short_title=CommandId.SHAPES_SELECT_ALL.title,
        callback=_shapes_actions.select_all_shapes,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_SELECT_ALL],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_DELETE,
        title=CommandId.SHAPES_DELETE.description,
        short_title=CommandId.SHAPES_DELETE.title,
        callback=_shapes_actions.delete_selected_shapes,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_DELETE],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_MOVE_TO_FRONT,
        title=CommandId.SHAPES_MOVE_TO_FRONT.title,
        callback=_shapes_actions.move_shapes_selection_to_front,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_MOVE_TO_FRONT],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_MOVE_TO_BACK,
        title=CommandId.SHAPES_MOVE_TO_BACK.title,
        callback=_shapes_actions.move_shapes_selection_to_back,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_MOVE_TO_BACK],
        enablement=enablement,
    ),
    Action(
        id=CommandId.SHAPES_FINISH_DRAWING_SHAPE,
        title=CommandId.SHAPES_FINISH_DRAWING_SHAPE.description,
        short_title=CommandId.SHAPES_FINISH_DRAWING_SHAPE.title,
        callback=_shapes_actions.finish_drawing_shape,
        keybindings=DEFAULT_SHORTCUTS[CommandId.SHAPES_FINISH_DRAWING_SHAPE],
        enablement=enablement,
    ),
]

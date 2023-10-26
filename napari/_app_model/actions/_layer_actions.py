"""This module defines actions (functions) that operate on layers.

Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.

The Actions in LAYER_ACTIONS are registered with the application when it is
created in `_app_model._app`.  Modifying this list at runtime will have no
effect.  Use `app.register_action` to register new actions at runtime.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, List

from app_model.types import Action

from napari._app_model.constants import CommandId, MenuGroup, MenuId
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
from napari.layers import _layer_actions

if TYPE_CHECKING:
    from app_model.types import MenuRuleDict

# The following dicts define groups to which menu items in the layer list context menu can belong
# see https://app-model.readthedocs.io/en/latest/types/#app_model.types.MenuRule for details
LAYERCTX_SPLITMERGE: MenuRuleDict = {
    'id': MenuId.LAYERLIST_CONTEXT,
    'group': MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
}
LAYERCTX_CONVERSION: MenuRuleDict = {
    'id': MenuId.LAYERLIST_CONTEXT,
    'group': MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
}
LAYERCTX_LINK: MenuRuleDict = {
    'id': MenuId.LAYERLIST_CONTEXT,
    'group': MenuGroup.LAYERLIST_CONTEXT.LINK,
}

# Statically defined Layer actions.
# modifying this list at runtime has no effect.
LAYER_ACTIONS: List[Action] = [
    Action(
        id=CommandId.LAYER_DUPLICATE,
        title=CommandId.LAYER_DUPLICATE.command_title,
        callback=_layer_actions._duplicate_layer,
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id=CommandId.LAYER_SPLIT_STACK,
        title=CommandId.LAYER_SPLIT_STACK.command_title,
        callback=_layer_actions._split_stack,
        menus=[{**LAYERCTX_SPLITMERGE, 'when': ~LLSCK.active_layer_is_rgb}],
        enablement=LLSCK.active_layer_is_image_3d,
    ),
    Action(
        id=CommandId.LAYER_SPLIT_RGB,
        title=CommandId.LAYER_SPLIT_RGB.command_title,
        callback=_layer_actions._split_rgb,
        menus=[{**LAYERCTX_SPLITMERGE, 'when': LLSCK.active_layer_is_rgb}],
        enablement=LLSCK.active_layer_is_rgb,
    ),
    Action(
        id=CommandId.LAYER_CONVERT_TO_LABELS,
        title=CommandId.LAYER_CONVERT_TO_LABELS.command_title,
        callback=_layer_actions._convert_to_labels,
        enablement=(
            (
                (LLSCK.num_selected_image_layers >= 1)
                | (LLSCK.num_selected_shapes_layers >= 1)
            )
            & LLSCK.all_selected_layers_same_type
            & ~LLSCK.selected_empty_shapes_layer
        ),
        menus=[LAYERCTX_CONVERSION],
    ),
    Action(
        id=CommandId.LAYER_CONVERT_TO_IMAGE,
        title=CommandId.LAYER_CONVERT_TO_IMAGE.command_title,
        callback=_layer_actions._convert_to_image,
        enablement=(
            (LLSCK.num_selected_labels_layers >= 1)
            & LLSCK.all_selected_layers_same_type
        ),
        menus=[LAYERCTX_CONVERSION],
    ),
    Action(
        id=CommandId.LAYER_MERGE_STACK,
        title=CommandId.LAYER_MERGE_STACK.command_title,
        callback=_layer_actions._merge_stack,
        enablement=(
            (LLSCK.num_selected_layers > 1)
            & (LLSCK.num_selected_image_layers == LLSCK.num_selected_layers)
            & LLSCK.all_selected_layers_same_shape
        ),
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id=CommandId.LAYER_TOGGLE_VISIBILITY,
        title=CommandId.LAYER_TOGGLE_VISIBILITY.command_title,
        callback=_layer_actions._toggle_visibility,
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id=CommandId.LAYER_LINK_SELECTED,
        title=CommandId.LAYER_LINK_SELECTED.command_title,
        callback=_layer_actions._link_selected_layers,
        enablement=(
            (LLSCK.num_selected_layers > 1) & ~LLSCK.num_selected_layers_linked
        ),
        menus=[{**LAYERCTX_LINK, 'when': ~LLSCK.num_selected_layers_linked}],
    ),
    Action(
        id=CommandId.LAYER_UNLINK_SELECTED,
        title=CommandId.LAYER_UNLINK_SELECTED.command_title,
        callback=_layer_actions._unlink_selected_layers,
        enablement=LLSCK.num_selected_layers_linked,
        menus=[{**LAYERCTX_LINK, 'when': LLSCK.num_selected_layers_linked}],
    ),
    Action(
        id=CommandId.LAYER_SELECT_LINKED,
        title=CommandId.LAYER_SELECT_LINKED.command_title,
        callback=_layer_actions._select_linked_layers,
        enablement=LLSCK.num_unselected_linked_layers,
        menus=[LAYERCTX_LINK],
    ),
    Action(
        id=CommandId.SHOW_SELECTED_LAYERS,
        title=CommandId.SHOW_SELECTED_LAYERS.command_title,
        callback=_layer_actions._show_selected,
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id=CommandId.HIDE_SELECTED_LAYERS,
        title=CommandId.HIDE_SELECTED_LAYERS.command_title,
        callback=_layer_actions._hide_selected,
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id=CommandId.SHOW_UNSELECTED_LAYERS,
        title=CommandId.SHOW_UNSELECTED_LAYERS.command_title,
        callback=_layer_actions._show_unselected,
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
    Action(
        id=CommandId.HIDE_UNSELECTED_LAYERS,
        title=CommandId.HIDE_UNSELECTED_LAYERS.command_title,
        callback=_layer_actions._hide_unselected,
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.NAVIGATION,
            }
        ],
    ),
]

for _dtype in (
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
):
    cmd = getattr(CommandId, f'LAYER_CONVERT_TO_{_dtype.upper()}')
    LAYER_ACTIONS.append(
        Action(
            id=cmd,
            title=cmd.command_title,
            callback=partial(_layer_actions._convert_dtype, mode=_dtype),
            enablement=(
                LLSCK.all_selected_layers_labels
                & (LLSCK.active_layer_dtype != _dtype)
            ),
            menus=[{'id': MenuId.LAYERS_CONVERT_DTYPE}],
        )
    )

for mode in ('max', 'min', 'std', 'sum', 'mean', 'median'):
    cmd = getattr(CommandId, f'LAYER_PROJECT_{mode.upper()}')
    LAYER_ACTIONS.append(
        Action(
            id=cmd,
            title=cmd.command_title,
            callback=partial(_layer_actions._project, mode=mode),
            enablement=LLSCK.active_layer_is_image_3d,
            menus=[{'id': MenuId.LAYERS_PROJECT}],
        )
    )

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
from napari._app_model.context import LayerListContextKeys as LLCK
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

_ONLY_LABELS = LLCK.num_selected_labels_layers == LLCK.num_selected_layers
_IMAGE_IS_3D = (LLCK.active_layer_type == "image") & LLCK.active_layer_ndim > 2


# Statically defined Layer actions.
# modifying this list at runtime has no effect.
LAYER_ACTIONS: List[Action] = [
    Action(
        id=CommandId.LAYER_DUPLICATE,
        title=CommandId.LAYER_DUPLICATE.title,
        callback=_layer_actions._duplicate_layer,
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id=CommandId.LAYER_SPLIT_STACK,
        title=CommandId.LAYER_SPLIT_STACK.title,
        callback=_layer_actions._split_stack,
        menus=[{**LAYERCTX_SPLITMERGE, 'when': ~LLCK.active_layer_is_rgb}],
        enablement=LLCK.active_layer_type == "image",
    ),
    Action(
        id=CommandId.LAYER_SPLIT_RGB,
        title=CommandId.LAYER_SPLIT_RGB.title,
        callback=_layer_actions._split_rgb,
        menus=[{**LAYERCTX_SPLITMERGE, 'when': LLCK.active_layer_is_rgb}],
        enablement=LLCK.active_layer_is_rgb,
    ),
    Action(
        id=CommandId.LAYER_CONVERT_TO_LABELS,
        title=CommandId.LAYER_CONVERT_TO_LABELS.title,
        callback=_layer_actions._convert_to_labels,
        enablement=(
            (
                (LLCK.num_selected_image_layers >= 1)
                | (LLCK.num_selected_shapes_layers >= 1)
            )
            & LLCK.all_selected_layers_same_type
        ),
        menus=[LAYERCTX_CONVERSION],
    ),
    Action(
        id=CommandId.LAYER_CONVERT_TO_IMAGE,
        title=CommandId.LAYER_CONVERT_TO_IMAGE.title,
        callback=_layer_actions._convert_to_image,
        enablement=(
            (LLCK.num_selected_labels_layers >= 1)
            & LLCK.all_selected_layers_same_type
        ),
        menus=[LAYERCTX_CONVERSION],
    ),
    Action(
        id=CommandId.LAYER_MERGE_STACK,
        title=CommandId.LAYER_MERGE_STACK.title,
        callback=_layer_actions._merge_stack,
        enablement=(
            (LLCK.num_selected_layers > 1)
            & (LLCK.num_selected_image_layers == LLCK.num_selected_layers)
            & LLCK.all_selected_layers_same_shape
        ),
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id=CommandId.LAYER_TOGGLE_VISIBILITY,
        title=CommandId.LAYER_TOGGLE_VISIBILITY.title,
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
        title=CommandId.LAYER_LINK_SELECTED.title,
        callback=_layer_actions._link_selected_layers,
        enablement=(
            (LLCK.num_selected_layers > 1) & ~LLCK.num_selected_layers_linked
        ),
        menus=[{**LAYERCTX_LINK, 'when': ~LLCK.num_selected_layers_linked}],
    ),
    Action(
        id=CommandId.LAYER_UNLINK_SELECTED,
        title=CommandId.LAYER_UNLINK_SELECTED.title,
        callback=_layer_actions._unlink_selected_layers,
        enablement=LLCK.num_selected_layers_linked,
        menus=[{**LAYERCTX_LINK, 'when': LLCK.num_selected_layers_linked}],
    ),
    Action(
        id=CommandId.LAYER_SELECT_LINKED,
        title=CommandId.LAYER_SELECT_LINKED.title,
        callback=_layer_actions._select_linked_layers,
        enablement=LLCK.num_unselected_linked_layers,
        menus=[LAYERCTX_LINK],
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
    cmd: CommandId = getattr(CommandId, f'LAYER_CONVERT_TO_{_dtype.upper()}')
    LAYER_ACTIONS.append(
        Action(
            id=cmd,
            title=cmd.title,
            callback=partial(_layer_actions._convert_dtype, mode=_dtype),
            enablement=(_ONLY_LABELS & (LLCK.active_layer_dtype != _dtype)),
            menus=[{'id': MenuId.LAYERS_CONVERT_DTYPE}],
        )
    )

for mode in ('max', 'min', 'std', 'sum', 'mean', 'median'):
    cmd: CommandId = getattr(CommandId, f'LAYER_PROJECT_{mode.upper()}')
    LAYER_ACTIONS.append(
        Action(
            id=cmd,
            title=cmd.title,
            callback=partial(_layer_actions._project, mode=mode),
            enablement=_IMAGE_IS_3D,
            menus=[{'id': MenuId.LAYERS_PROJECT}],
        )
    )

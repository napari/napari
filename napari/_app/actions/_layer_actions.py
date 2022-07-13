"""This module contains actions (functions) that operate on layers.

Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, List

from app_model.types import Action

from ...layers import _layer_actions
from ...utils.translations import trans
from .._menus import MenuGroup, MenuId
from ..context import LayerListContextKeys as LLCK

if TYPE_CHECKING:
    from app_model.types import MenuRuleDict


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

_only_labels = LLCK.num_selected_labels_layers == LLCK.num_selected_layers

# sourcery skip: for-append-to-extend
LAYER_ACTIONS: List[Action] = [
    Action(
        id='napari:layers:duplicate_layer',
        title=trans._('Duplicate Layer'),
        callback='napari.layers._layer_actions:_duplicate_layer',
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id='napari:split_stack',
        title=trans._('Split Stack'),
        callback='napari.layers._layer_actions:_split_stack',
        menus=[{**LAYERCTX_SPLITMERGE, 'when': ~LLCK.active_layer_is_rgb}],
        enablement=LLCK.active_layer_type == "image",
    ),
    Action(
        id='napari:split_rgb',
        title=trans._('Split RGB'),
        callback='napari.layers._layer_actions:_split_rgb',
        menus=[{**LAYERCTX_SPLITMERGE, 'when': LLCK.active_layer_is_rgb}],
        enablement=LLCK.active_layer_is_rgb,
    ),
    Action(
        id='napari:convert_to_labels',
        title=trans._('Convert to Labels'),
        callback='napari.layers._layer_actions:_convert_to_labels',
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
        id='napari:convert_to_image',
        title=trans._('Convert to Image'),
        callback='napari.layers._layer_actions:_convert_to_image',
        enablement=(
            (LLCK.num_selected_labels_layers >= 1)
            & LLCK.all_selected_layers_same_type
        ),
        menus=[LAYERCTX_CONVERSION],
    ),
    Action(
        id='napari:merge_stack',
        title=trans._('Merge to Stack'),
        callback='napari.layers._layer_actions:_merge_stack',
        enablement=(
            (LLCK.num_selected_layers > 1)
            & (LLCK.num_selected_image_layers == LLCK.num_selected_layers)
            & LLCK.all_selected_layers_same_shape
        ),
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id='napari:toggle_visibility',
        callback='napari.layers._layer_actions:_toggle_visibility',
        title=trans._('Toggle visibility'),
        menus=[
            {
                'id': MenuId.LAYERLIST_CONTEXT,
                'group': MenuGroup.LAYERLIST_CONTEXT.NAVIGATION,
            }
        ],
    ),
    Action(
        id='napari:link_selected_layers',
        title=trans._('Link Layers'),
        callback='napari.layers._layer_actions:_link_selected_layers',
        enablement=(
            (LLCK.num_selected_layers > 1) & ~LLCK.num_selected_layers_linked
        ),
        menus=[{**LAYERCTX_LINK, 'when': ~LLCK.num_selected_layers_linked}],
    ),
    Action(
        id='napari:unlink_selected_layers',
        title=trans._('Unlink Layers'),
        callback='napari.layers._layer_actions:_unlink_selected_layers',
        enablement=LLCK.num_selected_layers_linked,
        menus=[{**LAYERCTX_LINK, 'when': LLCK.num_selected_layers_linked}],
    ),
    Action(
        id='napari:select_linked_layers',
        callback='napari.layers._layer_actions:_select_linked_layers',
        title=trans._('Select Linked Layers'),
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
    LAYER_ACTIONS.append(
        Action(
            id=f'napari:convert_to_{_dtype}',
            title=trans._('Convert to {dtype}', dtype=_dtype),
            callback=partial(_layer_actions._convert_dtype, mode=_dtype),
            enablement=(_only_labels & (LLCK.active_layer_dtype != _dtype)),
            menus=[{'id': MenuId.LAYERS_CONVERT_DTYPE}],
        )
    )

_image_is_3d = (LLCK.active_layer_type == "image") & LLCK.active_layer_ndim > 2
for mode in ('max', 'min', 'std', 'sum', 'mean', 'median'):
    LAYER_ACTIONS.append(
        Action(
            id=f'napari:{mode}_projection',
            title=trans._('{mode} projection', mode=mode.title()),
            callback=partial(_layer_actions._project, mode=mode),
            enablement=_image_is_3d,
            menus=[{'id': MenuId.LAYERS_PROJECT}],
        )
    )

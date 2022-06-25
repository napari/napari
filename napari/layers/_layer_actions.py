"""This module contains actions (functions) that operate on layers.

Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, List, cast

import numpy as np

from ..utils._injection import inject_napari_dependencies
from ..utils.actions import Action, MenuGroup, MenuId, register_action
from ..utils.context._layerlist_context import LayerListContextKeys as LLCK
from ..utils.translations import trans
from . import Image, Labels, Layer
from .utils import stack_utils
from .utils._link_layers import get_linked_layers

if TYPE_CHECKING:
    from ..components import LayerList
    from ..utils.actions._types import MenuRuleDict


def _duplicate_layer(ll: LayerList, *, name: str = ''):
    from copy import deepcopy

    for lay in list(ll.selection):
        new = deepcopy(lay)
        new.name = name or f'{new.name} copy'
        ll.insert(ll.index(lay) + 1, new)


def _split_stack(ll: LayerList, axis: int = 0):
    layer = ll.selection.active
    if not isinstance(layer, Image):
        return
    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    ll.remove(layer)
    ll.extend(images)
    ll.selection = set(images)  # type: ignore


def _convert(ll: LayerList, type_: str):
    from ..layers import Shapes

    for lay in list(ll.selection):
        idx = ll.index(lay)
        ll.pop(idx)
        if isinstance(lay, Shapes) and type_ == 'labels':
            data = lay.to_labels()
        else:
            data = lay.data.astype(int) if type_ == 'labels' else lay.data
        new_layer = Layer.create(data, lay._get_base_state(), type_)
        ll.insert(idx, new_layer)


# TODO: currently, we have to create a thin _convert_to_x wrapper around _convert
# here for the purpose of type hinting (which partial doesn't do) ...
# so that inject_dependencies works correctly.
# however, we could conceivably add an `args` option to register_action
# that would allow us to pass additional arguments, like a partial.
def _convert_to_labels(ll: LayerList):
    return _convert(ll, 'labels')


def _convert_to_image(ll: LayerList):
    return _convert(ll, 'image')


def _merge_stack(ll: LayerList, rgb=False):
    # force selection to follow LayerList ordering
    imgs = cast(List[Image], [layer for layer in ll if layer in ll.selection])
    assert all(isinstance(layer, Image) for layer in imgs)
    merged = (
        stack_utils.merge_rgb(imgs)
        if rgb
        else stack_utils.images_to_stack(imgs)
    )
    for layer in imgs:
        ll.remove(layer)
    ll.append(merged)


def _toggle_visibility(ll: LayerList):
    for lay in ll.selection:
        lay.visible = not lay.visible


def _select_linked_layers(ll: LayerList):
    ll.selection.update(get_linked_layers(*ll.selection))


@inject_napari_dependencies
def _convert_dtype(ll: LayerList, mode='int64'):
    if not (layer := ll.selection.active):
        return

    if not isinstance(layer, Labels):
        raise NotImplementedError(
            trans._(
                "Data type conversion only implemented for labels",
                deferred=True,
            )
        )

    target_dtype = np.dtype(mode)
    if (
        np.min(layer.data) < np.iinfo(target_dtype).min
        or np.max(layer.data) > np.iinfo(target_dtype).max
    ):
        raise AssertionError(
            trans._(
                "Labeling contains values outside of the target data type range.",
                deferred=True,
            )
        )
    else:
        layer.data = layer.data.astype(np.dtype(mode))


def _project(ll: LayerList, axis: int = 0, mode='max'):
    layer = ll.selection.active
    if not layer:
        return
    if not isinstance(layer, Image):
        raise NotImplementedError(
            trans._(
                "Projections are only implemented for images", deferred=True
            )
        )

    # this is not the desired behavior for coordinate-based layers
    # but the action is currently only enabled for 'image_active and ndim > 2'
    # before opening up to other layer types, this line should be updated.
    data = (getattr(np, mode)(layer.data, axis=axis, keepdims=False),)

    # get the meta data of the layer, but without transforms
    meta = {
        key: layer._get_base_state()[key]
        for key in layer._get_base_state()
        if key not in ('scale', 'translate', 'rotate', 'shear', 'affine')
    }
    meta.update(  # sourcery skip
        {
            'name': f'{layer} {mode}-proj',
            'colormap': layer.colormap.name,
            'rendering': layer.rendering,
        }
    )
    new = Layer.create(data, meta, layer._type_string)
    # add transforms from original layer, but drop the axis of the projection
    new._transforms = layer._transforms.set_slice(
        [ax for ax in range(layer.ndim) if ax != axis]
    )

    ll.append(new)


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
        run=_duplicate_layer,
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id='napari:split_stack',
        title=trans._('Split Stack'),
        run=_split_stack,
        enablement=LLCK.active_layer_type == "image",
        menus=[{**LAYERCTX_SPLITMERGE, 'when': ~LLCK.active_layer_is_rgb}],
    ),
    Action(
        id='napari:split_stack',
        title=trans._('Split RGB'),
        run=_split_stack,
        menus=[{**LAYERCTX_SPLITMERGE, 'when': LLCK.active_layer_is_rgb}],
        enablement=LLCK.active_layer_is_rgb,
    ),
    Action(
        id='napari:convert_to_labels',
        title=trans._('Convert to Labels'),
        run=_convert_to_labels,
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
        run=_convert_to_image,
        enablement=(
            (LLCK.num_selected_labels_layers >= 1)
            & LLCK.all_selected_layers_same_type
        ),
        menus=[LAYERCTX_CONVERSION],
    ),
    Action(
        id='napari:merge_stack',
        title=trans._('Merge to Stack'),
        run=_merge_stack,
        enablement=(
            (LLCK.num_selected_layers > 1)
            & (LLCK.num_selected_image_layers == LLCK.num_selected_layers)
            & LLCK.all_selected_layers_same_shape
        ),
        menus=[LAYERCTX_SPLITMERGE],
    ),
    Action(
        id='napari:toggle_visibility',
        run=_toggle_visibility,
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
        run=lambda ll: ll.link_layers(ll.selection),
        enablement=(
            (LLCK.num_selected_layers > 1) & ~LLCK.num_selected_layers_linked
        ),
        menus=[{**LAYERCTX_LINK, 'when': ~LLCK.num_selected_layers_linked}],
    ),
    Action(
        id='napari:unlink_selected_layers',
        title=trans._('Unlink Layers'),
        run=lambda ll: ll.unlink_layers(ll.selection),
        enablement=LLCK.num_selected_layers_linked,
        menus=[{**LAYERCTX_LINK, 'when': LLCK.num_selected_layers_linked}],
    ),
    Action(
        id='napari:select_linked_layers',
        run=_select_linked_layers,
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
            run=partial(_convert_dtype, mode=_dtype),
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
            run=partial(_project, mode=mode),
            enablement=_image_is_3d,
            menus=[{'id': MenuId.LAYERS_PROJECT}],
        )
    )


def _register_layer_actions():
    for action in LAYER_ACTIONS:
        register_action(action)

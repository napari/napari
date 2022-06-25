"""This module contains actions (functions) that operate on layers.

Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np

from ..utils._injection import inject_napari_dependencies
from ..utils.actions import MenuGroup, MenuId, register_action
from ..utils.context._layerlist_context import LayerListContextKeys as LLCK
from ..utils.translations import trans
from . import Image, Layer
from .utils import stack_utils
from .utils._link_layers import get_linked_layers

if TYPE_CHECKING:
    from ..components import LayerList
    from ..utils.actions._types import MenuRuleDict

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


@register_action(
    'napari:layers:duplicate_layer',
    title=trans._('Duplicate Layer'),
    menus=[LAYERCTX_SPLITMERGE],
)
def _duplicate_layer(ll: LayerList, *, name: str = ''):
    from copy import deepcopy

    for lay in list(ll.selection):
        new = deepcopy(lay)
        new.name = name or f'{new.name} copy'
        ll.insert(ll.index(lay) + 1, new)


@register_action(
    'napari:split_stack',
    title=trans._('Split Stack'),
    precondition=LLCK.active_layer_type == "image",
    menus=[{**LAYERCTX_SPLITMERGE, 'when': ~LLCK.active_layer_is_rgb}],
)
@register_action(
    'napari:split_stack',
    title=trans._('Split RGB'),
    menus=[{**LAYERCTX_SPLITMERGE, 'when': LLCK.active_layer_is_rgb}],
    precondition=LLCK.active_layer_is_rgb,
)
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
@register_action(
    'napari:convert_to_image',
    title=trans._('Convert to Labels'),
    precondition=(
        (
            (LLCK.num_selected_image_layers >= 1)
            | (LLCK.num_selected_shapes_layers >= 1)
        )
        & LLCK.all_selected_layers_same_type
    ),
    menus=[LAYERCTX_CONVERSION],
)
def _convert_to_labels(ll: LayerList):
    return _convert(ll, 'labels')


@register_action(
    'napari:convert_to_image',
    title=trans._('Convert to Image'),
    precondition=(
        (LLCK.num_selected_labels_layers >= 1)
        & LLCK.all_selected_layers_same_type
    ),
    menus=[LAYERCTX_CONVERSION],
)
def _convert_to_image(ll: LayerList):
    return _convert(ll, 'image')


@register_action(
    'napari:merge_stack',
    title=trans._('Merge to Stack'),
    precondition=(
        (LLCK.num_selected_layers > 1)
        & (LLCK.num_selected_image_layers == LLCK.num_selected_layers)
        & LLCK.all_selected_layers_same_shape
    ),
    menus=[LAYERCTX_SPLITMERGE],
)
def _merge_stack(ll: LayerList, rgb=False):
    # force selection to follow LayerList ordering
    selection = [layer for layer in ll if layer in ll.selection]
    for layer in selection:
        ll.remove(layer)
    if rgb:
        new = stack_utils.merge_rgb(selection)
    else:
        new = stack_utils.images_to_stack(selection)
    ll.append(new)


@register_action(
    'napari:toggle_visibility',
    title=trans._('Toggle visibility'),
    menus=[
        {
            'id': MenuId.LAYERLIST_CONTEXT,
            'group': MenuGroup.LAYERLIST_CONTEXT.NAVIGATION,
        }
    ],
)
def _toggle_visibility(ll: LayerList):
    for lay in ll.selection:
        lay.visible = not lay.visible


@register_action(
    'napari:select_linked_layers',
    title=trans._('Select Linked Layers'),
    precondition=LLCK.num_unselected_linked_layers,
    menus=[LAYERCTX_LINK],
)
def _select_linked_layers(ll: LayerList):
    ll.selection.update(get_linked_layers(*ll.selection))


register_action(
    'napari:link_selected_layers',
    title=trans._('Link Layers'),
    precondition=(
        (LLCK.num_selected_layers > 1) & ~LLCK.num_selected_layers_linked
    ),
    menus=[{**LAYERCTX_LINK, 'when': ~LLCK.num_selected_layers_linked}],
    run=lambda ll: ll.link_layers(ll.selection),
)
register_action(
    'napari:unlink_selected_layers',
    title=trans._('Unlink Layers'),
    precondition=LLCK.num_selected_layers_linked,
    menus=[{**LAYERCTX_LINK, 'when': LLCK.num_selected_layers_linked}],
    run=lambda ll: ll.unlink_layers(ll.selection),
)


@inject_napari_dependencies
def _convert_dtype(ll: LayerList, mode='int64'):
    layer = ll.selection.active
    if not layer:
        return
    if layer._type_string != 'labels':
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


def _register_dtype_actions():
    for dtype in (
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
    ):
        register_action(
            f'napari:convert_to_{dtype}',
            title=trans._('Convert to {dtype}', dtype=dtype),
            run=partial(_convert_dtype, mode=dtype),
            precondition=(
                (LLCK.num_selected_labels_layers == LLCK.num_selected_layers)
                & (LLCK.active_layer_dtype != dtype)
            ),
            menus=[{'id': MenuId.LAYERS_CONVERT_DTYPE}],
        )


def _project(ll: LayerList, axis: int = 0, mode='max'):
    layer = ll.selection.active
    if not layer:
        return
    if layer._type_string != 'image':
        raise NotImplementedError(
            trans._(
                "Projections are only implemented for images", deferred=True
            )
        )

    # this is not the desired behavior for coordinate-based layers
    # but the action is currently only enabled for 'image_active and ndim > 2'
    # before opening up to other layer types, this line should be updated.
    data = (getattr(np, mode)(layer.data, axis=axis, keepdims=False),)
    layer = cast('Image', layer)  # noqa: F821
    # get the meta data of the layer, but without transforms
    meta = {
        key: layer._get_base_state()[key]
        for key in layer._get_base_state()
        if key not in ('scale', 'translate', 'rotate', 'shear', 'affine')
    }
    meta.update(
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


def _register_projection_actions():
    for mode in ('max', 'min', 'std', 'sum', 'mean', 'median'):
        register_action(
            f'napari:{mode}_projection',
            title=trans._('{mode} projection', mode=mode.title()),
            run=partial(_project, mode=mode),
            precondition=(
                (LLCK.active_layer_type == "image") & LLCK.active_layer_ndim
                > 2
            ),
            menus=[{'id': MenuId.LAYERS_PROJECT}],
        )


_register_dtype_actions()
_register_projection_actions()


# Each item in LAYER_ACTIONS will be added to the `QtActionContextMenu` created
# in _qt.containers._layer_delegate.LayerDelegate (i.e. they are options in the
# menu when you right-click on a layer in the layerlist.)
#
# variable names used in the `enable_when` and `show_when` expressions must be
# keys in the napari.components.layerlist.CONTEXT_KEYS dict.  If you need a new
# context paramameter, add a key key:value pair to the CONTEXT_KEYS dict.
#
# `action` must be a callable that accepts a single argument, an instance of
# `LayerList`.
#
# Please don't abuse "show_when".  For best UI, the menu should be roughly the
# same length all the time (just with various grayed out options).  `show_when`
# works best when there two adjacent actions with opposite `show_when`
# expressions.  See, e.g., 'link_selected_layers' and 'unlink_selected_layers'

# To add a separator, add any key with a value of _SEPARATOR

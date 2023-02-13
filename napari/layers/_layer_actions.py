"""This module contains actions (functions) that operate on layers.
Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, cast

import numpy as np

from napari.layers import Image, Labels, Layer
from napari.layers._source import layer_source
from napari.layers.utils import stack_utils
from napari.layers.utils._link_layers import get_linked_layers
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components import LayerList


def _duplicate_layer(ll: LayerList, *, name: str = ''):
    from copy import deepcopy

    for lay in list(ll.selection):
        data, state, type_str = lay.as_layer_data_tuple()
        state["name"] = trans._('{name} copy', name=lay.name)
        with layer_source(parent=lay):
            new = Layer.create(deepcopy(data), state, type_str)
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


def _split_rgb(ll: LayerList):
    return _split_stack(ll)


def _convert(ll: LayerList, type_: str):
    from napari.layers import Shapes

    float_dtypes = [np.float64, np.float32, np.float16]
    for lay in list(ll.selection):
        idx = ll.index(lay)
        ll.pop(idx)
        if isinstance(lay, Shapes) and type_ == 'labels':
            data = lay.to_labels()
        elif not issubclass(lay.data.dtype, np.integer) and type_ == 'labels':
            data = lay.data.astype(int)
        else:
            data = lay.data
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


def _link_selected_layers(ll: LayerList):
    ll.link_layers(ll.selection)


def _unlink_selected_layers(ll: LayerList):
    ll.unlink_layers(ll.selection)


def _select_linked_layers(ll: LayerList):
    ll.selection.update(get_linked_layers(*ll.selection))


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

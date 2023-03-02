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
    from napari.components import Dims, LayerList


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


def _project(ll: LayerList, dims: Dims, axis: int = 0, mode='max'):
    """Creates a new layer with specified projection.

    Parameters
    ----------
    ll : napari.componenets.LayerList
        The list of current layers in the viewer model.
    dims : napari.components.Dims
        The Dims model of the napari viewer with the current display order of the axes.
    axis : int
        The axis on which the values of the array will be projected. Note that this axis corresponds to the axis in
        the dims order, e.g if dims order is 2, 0, 1 and axis is 0, then axis 2 of the layer data will be used for the
        projection.
    mode : str
        Projection mode, either 'max', 'min', 'std', 'sum', 'mean', 'median'.
    """

    layer = ll.selection.active
    if not layer:
        return
    if not isinstance(layer, Image):
        raise NotImplementedError(
            trans._(
                "Projections are only implemented for images", deferred=True
            )
        )

    layer_data_order = list(range(len(layer.data.shape)))
    dims_order = list(dims.order)
    move_order = [dims_order.index(i) for i in layer_data_order]
    print(dims_order, move_order)
    data = getattr(np, mode)(
        np.moveaxis(layer.data, layer_data_order, move_order),
        axis=axis,
        keepdims=False,
    )
    # In case the last 2 dimensions are these axes, the data needs to be swapped in order to not have the projection
    # displayed orthogonal to the image data.
    must_swap = ((0, 1), (2, 1), (1, 0))
    if dims.order[1:] in must_swap:
        data = np.swapaxes(data, 0, 1)

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

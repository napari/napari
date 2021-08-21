"""This module contains actions (functions) that operate on layers.

Among other potential uses, these will populate the menu when you right-click
on a layer in the LayerList.
"""
from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Mapping,
    Sequence,
    Union,
    cast,
)

import numpy as np
from typing_extensions import TypedDict

from napari.experimental import link_layers, unlink_layers
from napari.layers.utils._link_layers import get_linked_layers

from ..utils.translations import trans
from .base.base import Layer
from .utils import stack_utils

if TYPE_CHECKING:
    from napari.components import LayerList
    from napari.layers import Image


def _duplicate_layer(ll: LayerList):
    from copy import deepcopy

    for lay in list(ll.selection):
        new = deepcopy(lay)
        new.name += ' copy'
        ll.insert(ll.index(lay) + 1, new)


def _split_stack(ll: LayerList, axis: int = 0):
    layer = ll.selection.active
    if not layer:
        return
    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    ll.remove(layer)
    ll.extend(images)
    ll.selection = set(images)  # type: ignore


def _project(ll: LayerList, axis: int = 0, mode='max'):
    layer = ll.selection.active
    if not layer:
        return
    if layer._type_string != 'image':
        raise NotImplementedError(
            "Projections are only implemented for images"
        )

    # this is not the desired behavior for coordinate-based layers
    # but the action is currently only enabled for 'image_active and ndim > 2'
    # before opening up to other layer types, this line should be updated.
    data = (getattr(np, mode)(layer.data, axis=axis, keepdims=True),)
    layer = cast('Image', layer)
    meta = {
        **layer._get_base_state(),
        'name': f'{layer} {mode}-proj',
        'colormap': layer.colormap.name,
        'interpolation': layer.interpolation,
        'rendering': layer.rendering,
    }
    new = Layer.create(data, meta, layer._type_string)
    ll.append(new)


def _convert(ll: LayerList, type_: str):

    for lay in list(ll.selection):
        idx = ll.index(lay)
        data = lay.data.astype(int) if type_ == 'labels' else lay.data
        ll.pop(idx)
        ll.insert(idx, Layer.create(data, {'name': lay.name}, type_))


def _merge_stack(ll: LayerList, rgb=False):
    selection = list(ll.selection)
    for layer in selection:
        ll.remove(layer)
    if rgb:
        new = stack_utils.merge_rgb(selection)
    else:
        new = stack_utils.images_to_stack(selection)
    ll.append(new)


def _select_linked_layers(ll: LayerList):
    ll.selection.update(get_linked_layers(*ll.selection))


class _MenuItem(TypedDict):
    """An object that encapsulates an Item in a QtActionContextMenu.

    Parameters
    ----------
    description : str
        The words that appear in the menu
    enable_when : str
        An expression that evaluates to a boolean (in namespace of some
        context) and controls whether the menu item is enabled.
    show_when : str
        An expression that evaluates to a boolean (in namespace of some
        context) and controls whether the menu item is visible.
    """

    description: str
    enable_when: str
    show_when: str


class ContextAction(_MenuItem):
    """An object that encapsulates a QAction in a QtActionContextMenu.

    Parameters
    ----------
    action : callable
        A function that may be called if the item is selected in the menu
    """

    action: Callable


class SubMenu(_MenuItem):
    action_group: Mapping[str, ContextAction]


MenuItem = Dict[str, Union[ContextAction, SubMenu]]

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


def _projdict(key) -> ContextAction:
    return {
        'description': key,
        'action': partial(_project, mode=key),
        'enable_when': 'image_active and ndim > 2',
        'show_when': 'True',
    }


_LAYER_ACTIONS: Sequence[MenuItem] = [
    {
        'napari:duplicate_layer': {
            'description': trans._('Duplicate Layer'),
            'action': _duplicate_layer,
            'enable_when': 'True',
            'show_when': 'True',
        },
        'napari:convert_to_labels': {
            'description': trans._('Convert to Labels'),
            'action': partial(_convert, type_='labels'),
            'enable_when': 'only_images_selected',
            'show_when': 'True',
        },
        'napari:convert_to_image': {
            'description': trans._('Convert to Image'),
            'action': partial(_convert, type_='image'),
            'enable_when': 'only_labels_selected',
            'show_when': 'True',
        },
    },
    # (each new dict creates a seperated section in the menu)
    {
        'napari:group:projections': {
            'description': trans._('Make Projection'),
            'enable_when': 'image_active and ndim > 2',
            'show_when': 'True',
            'action_group': {
                'napari:max_projection': _projdict('max'),
                'napari:min_projection': _projdict('min'),
                'napari:std_projection': _projdict('std'),
                'napari:sum_projection': _projdict('sum'),
                'napari:mean_projection': _projdict('mean'),
                'napari:median_projection': _projdict('median'),
            },
        }
    },
    {
        'napari:split_stack': {
            'description': trans._('Split Stack'),
            'action': _split_stack,
            'enable_when': 'image_active and active_layer_shape[0] < 10',
            'show_when': 'not active_is_rgb',
        },
        'napari:split_rgb': {
            'description': trans._('Split RGB'),
            'action': _split_stack,
            'enable_when': 'active_is_rgb',
            'show_when': 'active_is_rgb',
        },
        'napari:merge_stack': {
            'description': trans._('Merge to Stack'),
            'action': _merge_stack,
            'enable_when': (
                'selection_count > 1 and only_images_selected and same_shape'
            ),
            'show_when': 'True',
        },
    },
    {
        'napari:link_selected_layers': {
            'description': trans._('Link Layers'),
            'action': lambda ll: link_layers(ll.selection),
            'enable_when': 'selection_count > 1 and not all_layers_linked',
            'show_when': 'not all_layers_linked',
        },
        'napari:unlink_selected_layers': {
            'description': trans._('Unlink Layers'),
            'action': lambda ll: unlink_layers(ll.selection),
            'enable_when': 'all_layers_linked',
            'show_when': 'all_layers_linked',
        },
        'napari:select_linked_layers': {
            'description': trans._('Select Linked Layers'),
            'action': _select_linked_layers,
            'enable_when': 'linked_layers_unselected',
            'show_when': 'True',
        },
    },
]

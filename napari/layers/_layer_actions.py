from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional

from typing_extensions import TypedDict

from napari.experimental import link_layers, unlink_layers
from napari.layers.utils._link_layers import get_linked_layers

from .utils import stack_utils

if TYPE_CHECKING:
    from napari.components import LayerList


class ContextAction(TypedDict):
    description: str
    action: Callable
    enable_when: str
    show_when: Optional[str]


def _duplicate_layer(ll: LayerList):

    for lay in list(ll.selection):
        new = deepcopy(lay)
        new.name += ' copy'
        ll.insert(ll.index(lay) + 1, new)


def _split_stack(ll: LayerList, axis: int = 0):
    layer = ll.selection.active
    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    ll.remove(layer)
    ll.extend(images)
    ll.selection = set(images)


def _convert(ll: LayerList, type_: str):
    from .base.base import Layer

    for lay in list(ll.selection):
        idx = ll.index(lay)
        data = lay.data.astype(int) if type_ == 'labels' else lay.data
        ll.pop(idx)
        ll.insert(idx, Layer.create(data, {'name': lay.name}, type_))


def _merge_stack(layer_list, rgb=False):
    selection = list(layer_list.selection)
    for layer in selection:
        layer_list.remove(layer)
    if rgb:
        new = stack_utils.merge_rgb(selection)
    else:
        new = stack_utils.images_to_stack(selection)
    layer_list.append(new)


LAYER_ACTIONS = {
    'napari:duplicate_layer': {
        'description': 'Duplicate Layer',
        'action': _duplicate_layer,
        'enable_when': 'True',
    },
    'napari:convert_to_labels': {
        'description': 'Convert to Labels',
        'action': partial(_convert, type_='labels'),
        'enable_when': 'only_images_selected',
    },
    'napari:convert_to_image': {
        'description': 'Convert to Image',
        'action': partial(_convert, type_='image'),
        'enable_when': 'only_labels_selected',
    },
    'sep0': {},
    'napari:split_stack': {
        'description': 'Split Stack',
        'action': _split_stack,
        'enable_when': 'image_active and active_shape[0] < 10',
        'show_when': 'not active_is_rgb',
    },
    'napari:split_rgb': {
        'description': 'Split RGB',
        'action': _split_stack,
        'enable_when': 'active_is_rgb',
        'show_when': 'active_is_rgb',
    },
    'napari:merge_stack': {
        'description': 'Merge to Stack',
        'action': _merge_stack,
        'enable_when': 'only_images_selected and same_shape',
    },
    # 'napari:merge_to_rgb': {
    #     'description': 'Merge to RGB',
    #     'action': partial(merge_stack, rgb=True),
    #     'enable_when': 'only_images_selected and same_shape',
    # },
    'sep1': {},
    'napari:link_selected_layers': {
        'description': 'Link Layers',
        'action': lambda ll: link_layers(ll.selection),
        'enable_when': 'selection_count > 1 and not all_layers_linked',
        'show_when': 'not all_layers_linked',
    },
    'napari:unlink_selected_layers': {
        'description': 'Unlink Layers',
        'action': lambda ll: unlink_layers(ll.selection),
        'enable_when': 'all_layers_linked',
        'show_when': 'all_layers_linked',
    },
    'napari:select_linked_layers': {
        'description': 'Select Linked Layers',
        'action': lambda ll: (
            ll.selection.update(get_linked_layers(*ll.selection))
        ),
        'enable_when': 'linked_layers_unselected',
    },
}

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from ._context_keys import CtxKeys, RawContextKey

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari.utils.events import Selection

    LayerSel = Selection[Layer]
    OptInt = Optional[int]
    OptTupleInt = Optional[Tuple[int, ...]]


# defining these here rather than inline lambdas for the purpose of typing
def _len(s: LayerSel) -> int:
    return len(s)


def _all_linked(s: LayerSel) -> bool:
    from napari.layers.utils._link_layers import layer_is_linked

    return bool(s and all(layer_is_linked(x) for x in s))


def _n_unselected_links(s: LayerSel) -> int:
    from napari.layers.utils._link_layers import get_linked_layers

    return len(get_linked_layers(*s) - s)


def _is_rgb(s: LayerSel) -> bool:
    return getattr(s.active, "rgb", False)


def _only_img(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "image" for x in s))


def _only_labels(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "labels" for x in s))


def _img_active(s: LayerSel) -> bool:
    return bool(s.active and s.active._type_string == "Image")


def _active_ndim(s: LayerSel) -> OptInt:
    return s.active and getattr(s.active.data, "ndim", None)


def _active_shape(s: LayerSel) -> OptTupleInt:
    return s.active and getattr(s.active.data, "shape", None)


def _same_shape(s: LayerSel) -> bool:
    return len({getattr(x.data, "shape", ()) for x in s}) == 1


class LayerListContextKeys(CtxKeys):
    layers_selection_count = RawContextKey(
        "layers_selection_count",
        0,
        "Number of layers currently selected",
        _len,
    )
    all_layers_linked = RawContextKey(
        "all_layers_linked",
        False,
        "True when all selected layers are linked.",
        _all_linked,
    )
    unselected_linked_layers = RawContextKey(
        "unselected_linked_layers",
        0,
        "Number of unselected layers linked to selected layer(s)",
        _n_unselected_links,
    )
    active_layer_is_rgb = RawContextKey(
        "active_layer_is_rgb",
        False,
        "True when the active layer is RGB",
        _is_rgb,
    )
    only_images_selected = RawContextKey(
        "only_images_selected",
        False,
        "True when there is at least one selected layer and all selected layers are images",
        _only_img,
    )
    only_labels_selected = RawContextKey(
        "only_labels_selected",
        False,
        "True when there is at least one selected layer and all selected layers are labels",
        _only_labels,
    )
    active_layer_is_image = RawContextKey(
        "active_layer_is_image",
        False,
        "True when the active layer is an image",
        _img_active,
    )
    active_layer_ndim = RawContextKey['LayerSel', 'OptInt'](
        "active_layer_ndim",
        0,
        "Number of dimensions in the active layer, or None if nothing is active",
        _active_ndim,
    )
    active_layer_shape = RawContextKey['LayerSel', 'OptTupleInt'](
        "active_layer_shape",
        (),
        "Shape of the active layer, or None if nothing is active.",
        _active_shape,
    )
    all_layers_same_shape = RawContextKey(
        "all_layers_same_shape",
        False,
        "True when all selected layers have the same shape",
        _same_shape,
    )

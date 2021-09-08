from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from napari.utils.translations import trans

from ._context_keys import ContextNamespace, RawContextKey

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari.utils.events import Selection

    LayerSel = Selection[Layer]


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


def _active_type(s: LayerSel) -> Optional[str]:
    return s.active and s.active._type_string


def _active_ndim(s: LayerSel) -> Optional[int]:
    return s.active and getattr(s.active.data, "ndim", None)


def _active_shape(s: LayerSel) -> Optional[Tuple[int, ...]]:
    return s.active and getattr(s.active.data, "shape", None)


def _same_shape(s: LayerSel) -> bool:
    return len({getattr(x.data, "shape", ()) for x in s}) == 1


class LayerListContextKeys(ContextNamespace):
    layers_selection_count = RawContextKey(
        "layers_selection_count",
        0,
        trans._("Number of layers currently selected"),
        _len,
    )
    all_layers_linked = RawContextKey(
        "all_layers_linked",
        False,
        trans._("True when all selected layers are linked."),
        _all_linked,
    )
    unselected_linked_layers = RawContextKey(
        "unselected_linked_layers",
        0,
        trans._("Number of unselected layers linked to selected layer(s)"),
        _n_unselected_links,
    )
    active_layer_is_rgb = RawContextKey(
        "active_layer_is_rgb",
        False,
        trans._("True when the active layer is RGB"),
        _is_rgb,
    )
    active_layer_type = RawContextKey['LayerSel', Optional[str]](
        "active_layer_type",
        None,
        trans._(
            "Lowercase name of active layer type, or None of none active."
        ),
        _active_type,
    )
    only_images_selected = RawContextKey(
        "only_images_selected",
        False,
        trans._(
            "True when there is at least one selected layer and all selected layers are images"
        ),
        _only_img,
    )
    only_labels_selected = RawContextKey(
        "only_labels_selected",
        False,
        trans._(
            "True when there is at least one selected layer and all selected layers are labels"
        ),
        _only_labels,
    )
    active_layer_ndim = RawContextKey['LayerSel', Optional[int]](
        "active_layer_ndim",
        0,
        trans._(
            "Number of dimensions in the active layer, or `None` if nothing is active"
        ),
        _active_ndim,
    )
    active_layer_shape = RawContextKey['LayerSel', Optional[Tuple[int, ...]]](
        "active_layer_shape",
        (),
        trans._("Shape of the active layer, or `None` if nothing is active."),
        _active_shape,
    )
    all_layers_same_shape = RawContextKey(
        "all_layers_same_shape",
        False,
        trans._("True when all selected layers have the same shape"),
        _same_shape,
    )

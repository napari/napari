from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from ...utils._dtype import normalize_dtype
from ...utils.translations import trans
from ._context_keys import ContextKey, ContextNamespace

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ...layers import Layer
    from ...utils.events import Selection

    LayerSel = Selection[Layer]


def _len(s: LayerSel) -> int:
    return len(s)


def _all_linked(s: LayerSel) -> bool:
    from ...layers.utils._link_layers import layer_is_linked

    return bool(s and all(layer_is_linked(x) for x in s))


def _n_unselected_links(s: LayerSel) -> int:
    from ...layers.utils._link_layers import get_linked_layers

    return len(get_linked_layers(*s) - s)


def _is_rgb(s: LayerSel) -> bool:
    return getattr(s.active, "rgb", False)


def _only_img(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "image" for x in s))


def _only_labels(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "labels" for x in s))


def _active_type(s: LayerSel) -> Optional[str]:
    return s.active._type_string if s.active else None


def _active_ndim(s: LayerSel) -> Optional[int]:
    return getattr(s.active.data, "ndim", None) if s.active else None


def _active_shape(s: LayerSel) -> Optional[Tuple[int, ...]]:
    return getattr(s.active.data, "shape", None) if s.active else None


def _same_shape(s: LayerSel) -> bool:
    return len({getattr(x.data, "shape", ()) for x in s}) == 1


def _active_dtype(s: LayerSel) -> DTypeLike:
    dtype = None
    if s.active:
        try:
            dtype = normalize_dtype(s.active.data.dtype).__name__
        except AttributeError:
            pass
    return dtype


class LayerListContextKeys(ContextNamespace['LayerSel']):
    """These are the available context keys relating to a LayerList.

    along with default value, a description, and a function to retrieve the
    current value from layers.selection
    """

    layers_selection_count = ContextKey(
        0,
        trans._("Number of layers currently selected"),
        _len,
    )
    all_layers_linked = ContextKey(
        False,
        trans._("True when all selected layers are linked."),
        _all_linked,
    )
    unselected_linked_layers = ContextKey(
        0,
        trans._("Number of unselected layers linked to selected layer(s)"),
        _n_unselected_links,
    )
    active_layer_is_rgb = ContextKey(
        False,
        trans._("True when the active layer is RGB"),
        _is_rgb,
    )
    active_layer_type = ContextKey['LayerSel', Optional[str]](
        None,
        trans._(
            "Lowercase name of active layer type, or None of none active."
        ),
        _active_type,
    )
    only_images_selected = ContextKey(
        False,
        trans._(
            "True when there is at least one selected layer and all selected layers are images"
        ),
        _only_img,
    )
    only_labels_selected = ContextKey(
        False,
        trans._(
            "True when there is at least one selected layer and all selected layers are labels"
        ),
        _only_labels,
    )
    active_layer_ndim = ContextKey['LayerSel', Optional[int]](
        None,
        trans._(
            "Number of dimensions in the active layer, or `None` if nothing is active"
        ),
        _active_ndim,
    )
    active_layer_shape = ContextKey['LayerSel', Optional[Tuple[int, ...]]](
        (),
        trans._("Shape of the active layer, or `None` if nothing is active."),
        _active_shape,
    )
    active_layer_dtype = ContextKey(
        None,
        trans._("Dtype of the active layer, or `None` if nothing is active."),
        _active_dtype,
    )
    all_layers_same_shape = ContextKey(
        False,
        trans._("True when all selected layers have the same shape"),
        _same_shape,
    )

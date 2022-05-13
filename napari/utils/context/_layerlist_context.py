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


def _n_selected_imgs(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "image")


def _only_labels(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "labels" for x in s))


def _n_selected_labels(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "labels")


def _only_points(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "points" for x in s))


def _n_selected_points(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "labels")


def _only_shapes(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "shapes" for x in s))


def _n_selected_shapes(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "shapes")


def _only_surface(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "surface" for x in s))


def _n_selected_surfaces(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "surface")


def _only_vectors(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "vectors" for x in s))


def _n_selected_vectors(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "vectors")


def _only_tracks(s: LayerSel) -> bool:
    return bool(s and all(x._type_string == "tracks" for x in s))


def _n_selected_tracks(s: LayerSel) -> int:
    return sum(1 for x in s if x._type_string == "tracks")


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


def _same_dtype(s: LayerSel) -> bool:
    for idx, layer in enumerate(list(s)):
        next_layer = s[idx + 1] if idx < (len(s) - 1) else None
        if (type(layer) != type(next_layer)) and (next_layer is not None):
            return False
    return True


class LayerListContextKeys(ContextNamespace['LayerSel']):
    """These are the available context keys relating to a LayerList.

    along with default value, a description, and a function to retrieve the
    current value from layers.selection
    """

    num_selected_layers = ContextKey(
        0,
        trans._("Number of currently selected layers."),
        _len,
    )
    num_selected_layers_linked = ContextKey(
        False,
        trans._("True when all selected layers are linked."),
        _all_linked,
    )
    num_unselected_linked_layers = ContextKey(
        0,
        trans._("Number of unselected layers linked to selected layer(s)."),
        _n_unselected_links,
    )
    active_layer_is_rgb = ContextKey(
        False,
        trans._("True when the active layer is RGB."),
        _is_rgb,
    )
    active_layer_type = ContextKey['LayerSel', Optional[str]](
        None,
        trans._(
            "Lowercase name of active layer type, or None of none active."
        ),
        _active_type,
    )
    # TODO: try to reduce these `only_x_selected` to a single set of strings
    # or something... however, this would require that our context expressions
    # support Sets, tuples, lists, etc...  which they currently do not.
    num_selected_image_layers = ContextKey(
        0,
        trans._("Number of selected image layers."),
        _n_selected_imgs,
    )
    num_selected_labels_layers = ContextKey(
        0,
        trans._("Number of selected labels layers."),
        _n_selected_labels,
    )
    num_selected_points_layers = ContextKey(
        0,
        trans._("Number of selected points layers."),
        _n_selected_points,
    )
    num_selected_shapes_layers = ContextKey(
        0,
        trans._("Number of selected shapes layers."),
        _n_selected_shapes,
    )
    num_selected_surface_layers = ContextKey(
        0,
        trans._("Number of selected surface layers."),
        _n_selected_surfaces,
    )
    num_selected_vectors_layers = ContextKey(
        0,
        trans._("Number of selected vectors layers."),
        _n_selected_vectors,
    )
    num_selected_tracks_layers = ContextKey(
        0,
        trans._("Number of selected tracks layers."),
        _n_selected_tracks,
    )
    active_layer_ndim = ContextKey['LayerSel', Optional[int]](
        None,
        trans._(
            "Number of dimensions in the active layer, or `None` if nothing is active."
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
    all_selected_layers_same_shape = ContextKey(
        False,
        trans._("True when all selected layers have the same shape."),
        _same_shape,
    )

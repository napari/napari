from typing import Callable, Dict, Optional, Type, TypeVar

from ... import components, layers, viewer

T = TypeVar("T")
_NULL = object()


def _access_viewer() -> Optional[viewer.Viewer]:
    return get_accessor(viewer.Viewer)()


def _get_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := _access_viewer()) else None


def _get_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := _access_viewer()) else None


# registry of Type -> "accessor function"
# where each value is a function that is capable
# of retrieving an instance of its corresponding key type.
_ACCESSORS: Dict[Type, Callable[..., Optional[object]]] = {
    layers.Layer: _get_active_layer,
    viewer.Viewer: viewer.current_viewer,
    components.LayerList: _get_active_layer_list,
}


def get_accessor(type_: Type[T]) -> Optional[Callable[..., Optional[T]]]:
    """Return object accessor function given a type.

    An object accessor is a function that returns an instance of a
    particular object type. For example, given type `napari.Viewer`, we return
    a function that can be called to get the current viewer.

    This is a form of dependency injection, and, along with
    `inject_napari_dependencies`, allows us to inject current napari objects
    into functions based on type hints.
    """
    if type_ in _ACCESSORS:
        return _ACCESSORS[type_]

    if isinstance(type_, type):
        for key, val in _ACCESSORS.items():
            if issubclass(type_, key):
                return val  # type: ignore [return-type]
    return None


class set_accessor:
    """Set accessor(s) for given type(s).

    "Accessors" are functions that can retrieve an instance of a given type.
    For instance, `napari.viewer.current_viewer` is a function that can
    retrieve an instance of `napari.Viewer`.

    This is a class that behaves as a function or a context manager, that
    allows one to set an accessor function for a given type.

    Parameters
    ----------
    mapping : Dict[Type[T], Callable[..., Optional[T]]]
        a map of type -> accessor function, where each value is a function
        that is capable of retrieving an instance of the associated key/type.
    clobber : bool, optional
        Whether to override any existing accessor function, by default False.

    Raises
    ------
    ValueError
        if clobber is `False` and one of the keys in `mapping` is already
        registered.
    """

    def __init__(
        self, mapping: Dict[Type[T], Callable[..., Optional[T]]], clobber=False
    ):
        self._before = {}
        for k in mapping:
            if k in _ACCESSORS and not clobber:
                raise ValueError(
                    f"Class {k} already has an accessor and clobber is False"
                )
            self._before[k] = _ACCESSORS.get(k, _NULL)
        _ACCESSORS.update(mapping)

    def __enter__(self):
        return None

    def __exit__(self, *_):
        for key, val in self._before.items():
            if val is _NULL:
                del _ACCESSORS[key]
            else:
                _ACCESSORS[key] = val

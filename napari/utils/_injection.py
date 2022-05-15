from functools import wraps
from inspect import isgeneratorfunction, signature
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from typing_extensions import get_type_hints

from .. import components, layers, viewer
from ..viewer import current_viewer

T = TypeVar("T")
_NULL = object()


def _get_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := current_viewer()) else None


def _get_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := current_viewer()) else None


# registry of Type -> "accessor function"
# where each value is a function that is capable
# of retrieving an instance of it's corresponding key type.
_ACCESSORS: Dict[Type, Callable[..., Optional[object]]] = {
    layers.Layer: _get_active_layer,
    viewer.Viewer: current_viewer,
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
            if issubclass(type, key):
                return val  # type: ignore [return-type]
    return None


class set_accessor:
    """Set accessor(s) for given type(s).

    "Acessors" are functions that can retrieve an instance of a given type.
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
        Whether to override any existing accessor function, by default True.

    Raises
    ------
    ValueError
        if clobber is `True` and one of the keys in `mapping` is already
        registered.
    """

    def __init__(
        self, mapping: Dict[Type[T], Callable[..., Optional[T]]], clobber=True
    ):
        self._before = {}
        for k, v in mapping.items():
            if k in _ACCESSORS and clobber is False:
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


def napari_type_hints(obj: Any) -> Dict[str, Any]:
    import napari

    return get_type_hints(
        obj,
        {
            'napari': napari,
            **viewer.__dict__,
            **layers.__dict__,
            **components.__dict__,
        },
    )


def inject_napari_dependencies(func: Callable) -> Callable:
    """Create callable that can access napari objects based on type hints.

    This is form of dependency injection.  If a function includes a parameter
    that has a recognized napari type (e.g. `Viewer`, or `Layer`), then this
    function will return a new version of the input function that can be called
    *without* that particular parameter.

    Examples
    --------

    >>> def f(viewer: Viewer): ...
    >>> inspect.signature(f)
    <Signature (x: 'Viewer')>
    >>> f2 = inject_napari_dependencies(f)
    >>> inspect.signature(f2)
    <Signature (x: typing.Optional[napari.viewer.Viewer] = None)>
    # if f2 is called without x, the current_viewer will be provided for x


    Parameters
    ----------
    func : Callable
        A function with napari type hints.

    Returns
    -------
    Callable
        A function with napari dependencies injected
    """
    if not func.__code__.co_argcount:
        return func

    sig = signature(func)
    # get type hints for the object, with forward refs of napari hints resolved
    hints = napari_type_hints(func)
    # get accessor functions for each required parameter
    required = {}
    for name, hint in hints.items():
        if sig.parameters[name].default is sig.empty:
            required[name] = hint

    @wraps(func)
    def _exec(*args, **kwargs):
        # when we call the function, we call the accessor functions to get
        # the current napari objects
        _kwargs = {}
        for n, hint in required.items():
            if accessor := get_accessor(hint):
                _kwargs[n] = accessor()

        # but we use bind_partial to allow the caller to still provide
        # their own objects if desired.
        # (i.e. the injected deps are only used if needed)
        _kwargs.update(**sig.bind_partial(*args, **kwargs).arguments)
        return func(**_kwargs)

    out = _exec

    # if it came in as a generatorfunction, it needs to go out as one.
    if isgeneratorfunction(func):

        @wraps(func)
        def _gexec(*args, **kwargs):
            yield from _exec(*args, **kwargs)

        out = _gexec

    # update the signature
    p = [
        p.replace(default=None, annotation=Optional[hints[p.name]])
        if p.name in required
        else p
        for p in sig.parameters.values()
    ]
    out.__signature__ = sig.replace(parameters=p)
    return out

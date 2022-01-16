from functools import wraps
from inspect import isgeneratorfunction, signature
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar

from typing_extensions import get_type_hints

if TYPE_CHECKING:
    from ..layers import Layer

T = TypeVar("T")


def _get_active_layer() -> Optional['Layer']:
    from ..viewer import current_viewer

    viewer = current_viewer()
    return viewer.layers.selection.active if viewer else None


def get_accessor(type_: Type[T]) -> Optional[Callable[[], Optional[T]]]:
    """Return object accessor function given a type.

    An object accessor is a function that returns an instance of a
    particular object type. For example, given type `napari.Viewer`, we return
    a function that can be called to get the current viewer.

    This is a form of dependency injection, and, along with
    `inject_napari_dependencies`, allows us to inject current napari objects
    into functions based on type hints.
    """
    from ..layers import Layer
    from ..viewer import Viewer, current_viewer

    if isinstance(type_, type) and issubclass(type_, Layer):
        return _get_active_layer
    if isinstance(type_, type) and issubclass(type_, Viewer):
        return current_viewer


def napari_type_hints(obj: Any) -> Dict[str, Any]:
    import napari

    from .. import layers, viewer

    return get_type_hints(
        obj, {'napari': napari, **viewer.__dict__, **layers.__dict__}
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
    accessors = {}
    for name, hint in hints.items():
        if sig.parameters[name].default is sig.empty:
            accessor = get_accessor(hint)
            if accessor:
                accessors[name] = accessor

    @wraps(func)
    def _exec(*args, **kwargs):
        # when we call the function, we call the accessor functions to get
        # the current napari objects
        _kwargs = {k: accessor() for k, accessor in accessors.items()}
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
        if p.name in accessors and p.default is p.empty
        else p
        for p in sig.parameters.values()
    ]
    out.__signature__ = sig.replace(parameters=p)
    return out

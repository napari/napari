from functools import wraps
from inspect import Signature, isgeneratorfunction, signature
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from typing_extensions import get_type_hints

from ... import components, layers, viewer
from ._processors import get_processor, set_processors
from ._providers import get_provider, provider, set_providers

T = TypeVar("T")
_NULL = object()


__all__ = [
    'provider',
    'get_provider',
    'get_processor',
    'inject_napari_dependencies',
    'napari_type_hints',
    'set_providers',
    'set_processors',
]


def resolve_sig_hints(sig: Signature) -> Tuple[Any, Dict[str, Any]]:
    from typing import _eval_type

    import napari

    localns = {
        'napari': napari,
        **viewer.__dict__,
        **layers.__dict__,
        **components.__dict__,
    }

    required_parameters = {}
    for param in sig.parameters.values():
        if param.default is sig.empty:
            hint = _eval_type(param.annotation, {}, localns)
            required_parameters[param.name] = hint

    return_hint = _eval_type(sig.return_annotation, {}, localns)
    return return_hint, required_parameters


def inject_napari_dependencies(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator returns func that can access/process napari objects based on type hints.

    This is form of dependency injection, and result processing.  It does 2 things:

    1. If `func` includes a parameter that has a type with a registered provider
    (e.g. `Viewer`, or `Layer`), then this decorator will return a new version of
    the input function that can be called *without* that particular parameter.

    2. If `func` has a return type with a registered processor (e.g. `ImageData`),
    then this decorator will return a new version of the input function that, when
    called, will have the result automatically processed by the current processor
    for that type (e.g. in the case of `ImageData`, it will be added to the viewer.)

    Examples
    --------
    >>> def f(viewer: Viewer): ...
    >>> inspect.signature(f)
    <Signature (x: 'Viewer')>
    >>> f2 = inject_napari_dependencies(f)
    >>> inspect.signature(f2)
    <Signature (x: typing.Optional[napari.Viewer] = None)>
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
    if not func.__code__.co_argcount and 'return' not in getattr(
        func, '__annotations__', {}
    ):
        return func

    sig = signature(func)
    # get type hints for the object, with forward refs of napari hints resolved
    return_hint, required_parameters = resolve_sig_hints(sig)

    # get provider functions for each required parameter
    @wraps(func)
    def _exec(*args, **kwargs):
        # when we call the function, we call the provider functions to get
        # the current napari objects
        _kwargs = {}
        for n, hint in required_parameters.items():
            if provider := get_provider(hint):
                _kwargs[n] = provider()

        # but we use bind_partial to allow the caller to still provide
        # their own objects if desired.
        # (i.e. the injected deps are only used if needed)
        _kwargs.update(**sig.bind_partial(*args, **kwargs).arguments)
        result = func(**_kwargs)
        if return_hint and (processor := get_processor(return_hint)):
            processor(result)
        return result

    out = _exec

    # if it came in as a generatorfunction, it needs to go out as one.
    if isgeneratorfunction(func):

        @wraps(func)
        def _gexec(*args, **kwargs):
            yield from _exec(*args, **kwargs)

        out = _gexec

    # update the signature
    p = [
        p.replace(
            default=None, annotation=Optional[required_parameters[p.name]]
        )
        if p.name in required_parameters
        else p
        for p in sig.parameters.values()
    ]
    out.__signature__ = sig.replace(parameters=p)
    return out

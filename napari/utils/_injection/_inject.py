from __future__ import annotations

import warnings
from functools import wraps
from inspect import isgeneratorfunction
from typing import TYPE_CHECKING, cast

from ..translations import trans
from ._processors import get_processor
from ._providers import get_provider
from ._type_resolution import type_resolved_signature

if TYPE_CHECKING:
    from inspect import Signature
    from typing import Callable, Literal, Optional, TypeVar

    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")
    RaiseWarnReturnIgnore = Literal['raise', 'warn', 'return', 'ignore']


def inject_napari_dependencies(
    func: Callable[P, R],
    *,
    localns: Optional[dict] = None,
    on_unresolved_required_args: RaiseWarnReturnIgnore = 'raise',
    on_unannotated_required_args: RaiseWarnReturnIgnore = 'warn',
) -> Callable[P, R]:
    """Decorator returns func that can access/process napari objects based on type hints.

    This is form of dependency injection, and result processing.  It does 2 things:

    1. If `func` includes a parameter that has a type with a registered provider
    (e.g. `Viewer`, or `Layer`), then this decorator will return a new version of
    the input function that can be called *without* that particular parameter.

    2. If `func` has a return type with a registered processor (e.g. `ImageData`),
    then this decorator will return a new version of the input function that, when
    called, will have the result automatically processed by the current processor
    for that type (e.g. in the case of `ImageData`, it will be added to the viewer.)

    Parameters
    ----------
    func : Callable
        A function with napari type hints.

    Returns
    -------
    Callable
        A function with napari dependencies injected

    Examples
    --------
    >>> def f(viewer: 'Viewer'): return viewer
    >>> f2 = inject_napari_dependencies(f)
    # if f2 is called without x, the current_viewer will be provided for x
    >>> viewer = napari.Viewer()
    >>> assert f2() is viewer
    """
    # if the function takes no arguments and has no return annotation
    # there's nothing to be done
    if not func.__code__.co_argcount and 'return' not in getattr(
        func, '__annotations__', {}
    ):
        return func

    # get a signature object with all type annotations resolved
    # this may result in a NameError if a required argument is unresolveable.
    # There may also be unannotated required arguments, which will likely fail
    # when the function is called later. We break this out into a seperate
    # function to handle notifying the user on these cases.
    sig = _resolve_sig_or_inform(
        func,
        localns,
        on_unresolved_required_args,
        on_unannotated_required_args,
    )
    if sig is None:  # something went wrong, and the user was notified.
        return func
    process_return = sig.return_annotation is not sig.empty

    # get provider functions for each required parameter
    @wraps(func)
    def _exec(*args: P.args, **kwargs: P.kwargs) -> R:
        # we're actually calling the "injected function" now

        _sig = cast(Signature, sig)
        # first, get and call the provider functions for each parameter type:
        _kwargs = {
            param_name: provider()
            for param_name, param in _sig.parameters.items()
            if (provider := get_provider(param.annotation))
        }

        # use bind_partial to allow the caller to still provide their own arguments
        # if desired. (i.e. the injected deps are only used if not provided)
        bound = _sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        _kwargs.update(**bound.arguments)

        try:  # call the function with injected values
            result = func(**_kwargs)  # type: ignore [arg-type]
        except TypeError as e:
            # likely a required argument is still missing.
            raise TypeError(
                f'After injecting dependencies for arguments {set(_kwargs)}, {e}'
            ) from e

        if process_return and (
            processor := get_processor(_sig.return_annotation)
        ):
            processor(result)

        return result

    out = _exec

    # if it came in as a generatorfunction, it needs to go out as one.
    if isgeneratorfunction(func):

        @wraps(func)
        def _gexec(*args, **kwargs):
            yield from _exec(*args, **kwargs)  # type: ignore [misc]

        out = _gexec

    # update some metadata on the decorated function.
    out.__signature__ = sig  # type: ignore [attr-defined]
    out.__annotations__ = {
        **{p.name: p.annotation for p in sig.parameters.values()},
        "return": sig.return_annotation,
    }
    out.__doc__ = (
        out.__doc__ or ''
    ) + '\n\n*This function will inject napari dependencies when called.*'
    out._dependencies_injected = True  # type: ignore [attr-defined]
    return out


def _resolve_sig_or_inform(
    func: Callable,
    localns: Optional[dict],
    on_unresolved_required_args: bool,
    on_unannotated_required_args: bool,
) -> Optional[Signature]:
    """Helper function for user warnings/errors during inject_napari_dependencies.

    all parameters are described above in inject_napari_dependencies
    """
    try:
        sig = type_resolved_signature(
            func,
            localns=localns,
            raise_unresolved_optional_args=False,
            inject_napari_namespace=True,
        )
    except NameError as e:
        errmsg = str(e)
        if on_unresolved_required_args == 'raise':
            msg = trans._(
                '{errmsg}. To simply return the original function, pass `on_unresolved_required_args="return"`. To emit a warning, pass "warn".',
                deferred=True,
                errmsg=errmsg,
            )
            raise NameError(msg) from e
        if on_unresolved_required_args == 'warn':
            msg = trans._(
                '{errmsg}. To suppress this warning and simply return the original function, pass `on_unresolved_required_args="return"`.',
                deferred=True,
                errmsg=errmsg,
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        return None

    for param in sig.parameters.values():
        if param.default is param.empty and param.annotation is param.empty:
            base = trans._(
                'Injecting dependencies on function {fname!r} with a required, unannotated parameter {name!r}. This will fail later unless that parameter is provided at call-time.',
                name=param.name,
                fname=getattr(func, '__name__', ''),
            )
            if on_unannotated_required_args == 'raise':
                msg = trans._(
                    f'{base} To allow this, pass `on_unannotated_required_args="ignore"`. To emit a warning, pass "warn".',
                    base=base,
                    deferred=True,
                )
                raise TypeError(msg)
            elif on_unannotated_required_args == 'warn':
                msg = trans._(
                    f'{base} To allow this, pass `on_unannotated_required_args="ignore"`. To raise an exception, pass "raise".',
                    base=base,
                    deferred=True,
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            elif on_unannotated_required_args == 'return':
                return None

    return sig

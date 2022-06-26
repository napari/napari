from __future__ import annotations

import sys
import types
import typing
from functools import lru_cache
from inspect import Signature
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from typing import _get_type_hints_obj_allowed_types

PY39_OR_GREATER = sys.version_info >= (3, 9)


@lru_cache(maxsize=1)
def _napari_names() -> Dict[str, Any]:
    """Napari names to inject into local namespace when evaluating type hints."""
    import napari
    from napari import components, layers, viewer

    return {
        'napari': napari,
        **viewer.__dict__,
        **layers.__dict__,
        **components.__dict__,
    }


@lru_cache(maxsize=1)
def _typing_names():
    return {**typing.__dict__, **types.__dict__}  # noqa: TYP006


def resolve_type_hints(
    obj: _get_type_hints_obj_allowed_types,
    globalns: Optional[dict] = None,
    localns: Optional[dict] = None,
    include_extras: bool = False,
    inject_napari_namespace: bool = True,
) -> Dict[str, Any]:
    """Return type hints for an object.

    This is a small wrapper around `typing.get_type_hints()` that adds napari
    namespaces to the global and local namespaces.

    see docstring for :func:`typing.get_type_hints`.

    Parameters
    ----------
    obj : module, class, method, or function
        must be a module, class, method, or function.
    globalns : Optional[dict]
        optional global namespace, by default None.
    localns : Optional[dict]
        optional local namespace, by default None.
    include_extras : bool
        If `False` (the default), recursively replaces all 'Annotated[T, ...]'
        with 'T'.
    inject_napari_namespace : bool
        If `True` (the default), will inject all names from
        `napari.components`, `napari.layers`, and `napari.viewer`
        into the namespace when evaluating types.

    Returns
    -------
    Dict[str, Any]
        mapping of object name to type hint for all annotated attributes of `obj`.

    Examples
    --------
    >>> def func(image: 'Image', layer_list: 'LayerList') -> 'Labels': ...
    >>> resolve_type_hints(func)
    {
        'image': <class 'napari.layers.image.image.Image'>,
        'layer_list': <class 'napari.components.layerlist.LayerList'>,
        'return': <class 'napari.layers.labels.labels.Labels'>
    }
    """
    _localns = dict(_typing_names())
    if inject_napari_namespace:
        _localns.update(_napari_names())
    if localns:
        _localns.update(localns)  # explicitly provided locals take precedence
    return typing.get_type_hints(
        obj, globalns=globalns, localns=_localns, include_extras=include_extras
    )


def resolve_single_type_hints(
    *objs: Any,
    localns: Optional[dict] = None,
    include_extras: bool = False,
    inject_napari_namespace: bool = True,
) -> Tuple[Any, ...]:
    """Get type hints for one or more isolated type annotations.

    Wrapper around :func:`resolve_type_hints` (see docstring for that function for
    parameter docs).

    `typing.get_type_hints()` only works for modules, classes, methods, or functions,
    but the typing module doesn't make the underlying type evaluation logic publicly
    available. This function creates a small mock object with an `__annotations__`
    dict that will work as an argument to `typing.get_type_hints()`.  It then extracts
    the resolved hints back into a tuple of hints corresponding to the input objects.

    Returns
    -------
    Tuple[Any, ...]
        Tuple

    Examples
    --------
    >>> resolve_single_type_hints(int, 'Optional[int]', 'Viewer', 'LayerList')
    (
        <class 'int'>,
        typing.Optional[int],
        <class 'napari.viewer.Viewer'>,
        <class 'napari.components.layerlist.LayerList'>
    )

    >>> resolve_single_type_hints('hi', localns={'hi': typing.Any})
    (typing.Any,)
    """
    kwargs = dict(
        localns=localns,
        inject_napari_namespace=inject_napari_namespace,
    )
    if PY39_OR_GREATER:
        kwargs['include_extras'] = include_extras

    annotations = {str(n): v for n, v in enumerate(objs)}
    mock_obj = type('_T', (), {'__annotations__': annotations})()
    hints = resolve_type_hints(mock_obj, **kwargs)
    return tuple(hints[k] for k in annotations)


def type_resolved_signature(
    func: Callable,
    *,
    localns: Optional[dict] = None,
    raise_unresolved_optional_args: bool = True,
    inject_napari_namespace: bool = True,
) -> Signature:
    """Return a Signature object for a function with resolved type annotations.

    Parameters
    ----------
    func : Callable
        A callable object.
    localns : Optional[dict]
        Optional local namespace for name resolution, by default None
    raise_unresolved_optional_args : bool
        Whether to raise an exception when an optional parameter (one with a default
        value) has an unresolvable type annotation, by default True
    inject_napari_namespace : bool
        If `True` (the default), will inject all names from
        `napari.components`, `napari.layers`, and `napari.viewer`
        into the namespace when evaluating types.

    Returns
    -------
    Signature
        :class:`inspect.Signature` object with fully resolved type annotations,
        (or at least partially resolved type annotations if
        `raise_unresolved_optional_args` is `False`).

    Raises
    ------
    NameError
        If a required argument has an unresolvable type annotation, or if
        `raise_unresolved_optional_args` is `True` and an optional argument has
        an unresolvable type annotation.
    """
    sig = Signature.from_callable(func)
    try:
        hints = resolve_type_hints(
            func,
            localns=localns,
            inject_napari_namespace=inject_napari_namespace,
        )
    except NameError as err:
        if raise_unresolved_optional_args:
            raise NameError(
                f'Could not resolve all annotations in signature {sig} ({err}). '
                'To allow optional parameters and return types to remain unresolved, '
                'use `raise_unresolved_optional_args=False`'
            ) from err
        hints = _resolve_mandatory_params(sig, inject_napari_namespace)

    resolved_parameters = [
        param.replace(annotation=hints.get(param.name, param.empty))
        for param in sig.parameters.values()
    ]
    return sig.replace(
        parameters=resolved_parameters,
        return_annotation=hints.get('return', sig.empty),
    )


def _resolve_mandatory_params(
    sig: Signature,
    inject_napari_namespace: bool,
    exclude_unresolved_optionals: bool = False,
) -> Dict[str, Any]:
    """Resolve all required param annotations in `sig`, but allow optional ones to fail.

    Helper function for :func:`type_resolved_signature`.  This fallback function is
    used if at least one parameter in `sig` has an unresolvable type annotation.
    It resolves each parameter's type annotation independently, and only raises an
    error if a parameter without a default value has an unresolvable type annotation.

    If `exclude_unresolved_optionals` is `True`, then unresolved optional parameters
    will not appear in the output dict

    Returns
    -------
    Dict[str, Any]
        mapping of parameter name to type hint.

    Raises
    ------
    NameError
        If a required argument has an unresolvable type annotation.
    """
    hints = {}
    for name, param in sig.parameters.items():
        if param.annotation is sig.empty:
            continue  # pragma: no cover
        try:
            hints[name] = resolve_single_type_hints(
                param.annotation,
                inject_napari_namespace=inject_napari_namespace,
            )[0]
        except NameError as e:
            if param.default is param.empty:
                raise NameError(
                    f'Could not resolve type hint for required parameter {name!r}: {e}'
                ) from e
            elif not exclude_unresolved_optionals:
                hints[name] = param.annotation
    if sig.return_annotation is not sig.empty:
        try:
            hints['return'] = resolve_single_type_hints(
                sig.return_annotation,
                inject_napari_namespace=inject_napari_namespace,
            )[0]
        except NameError:
            if not exclude_unresolved_optionals:
                hints['return'] = sig.return_annotation
    return hints

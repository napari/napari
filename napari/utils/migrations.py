import warnings
from functools import wraps
from typing import Any, Callable

from napari.utils.translations import trans

_UNSET = object()


def rename_argument(
    from_name: str, to_name: str, version: str, since_version: str = ""
) -> Callable:
    """
    This is decorator for simple rename function argument
    without break backward compatibility.

    Parameters
    ----------
    from_name : str
        old name of argument
    to_name : str
        new name of argument
    version : str
        version when old argument will be removed
    since_version : str
        version when new argument was added
    """

    if not since_version:
        since_version = "unknown"
        warnings.warn(
            trans._(
                "The since_version argument was added in napari 0.4.18 and will be mandatory since 0.6.0 release.",
                deferred=True,
            ),
            stacklevel=2,
            category=FutureWarning,
        )

    def _wrapper(func):
        @wraps(func)
        def _update_from_dict(*args, **kwargs):
            if from_name in kwargs:
                if to_name in kwargs:
                    raise ValueError(
                        trans._(
                            "Argument {to_name} already defined, please do not mix {from_name} and {to_name} in one call.",
                            from_name=from_name,
                            to_name=to_name,
                        )
                    )
                warnings.warn(
                    trans._(
                        "Argument {from_name!r} is deprecated, please use {to_name!r} instead. The argument {from_name!r} was deprecated in {since_version} and it will be removed in {version}.",
                        from_name=from_name,
                        to_name=to_name,
                        version=version,
                        since_version=since_version,
                    ),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                kwargs = kwargs.copy()
                kwargs[to_name] = kwargs.pop(from_name)
            return func(*args, **kwargs)

        return _update_from_dict

    return _wrapper


def add_deprecated_property(
    obj: Any,
    previous_name: str,
    new_name: str,
    version: str,
    since_version: str,
) -> None:
    """
    Adds deprecated property and links to new property name setter and getter.

    Parameters
    ----------
    obj:
        Class instances to add property
    previous_name : str
        Name of previous property, its methods must be removed.
    new_name : str
        Name of new property, must have its getter (and setter if applicable) implemented.
    version : str
        Version where deprecated property will be removed.
    since_version : str
        version when new property was added
    """

    if hasattr(obj, previous_name):
        raise RuntimeError(
            trans._(
                "{previous_name} property already exists.",
                deferred=True,
                previous_name=previous_name,
            )
        )

    if not hasattr(obj, new_name):
        raise RuntimeError(
            trans._(
                "{new_name} property must exist.",
                deferred=True,
                new_name=new_name,
            )
        )

    name = f"{obj.__name__}.{previous_name}"
    msg = trans._(
        "{name} is deprecated since {since_version} and will be removed in {version}. Please use {new_name}",
        deferred=True,
        name=name,
        since_version=since_version,
        version=version,
        new_name=new_name,
    )

    def _getter(instance) -> Any:
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
        return getattr(instance, new_name)

    def _setter(instance, value: Any) -> None:
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
        setattr(instance, new_name, value)

    setattr(obj, previous_name, property(_getter, _setter))


def deprecated_constructor_arg_by_attr(name: str) -> Callable:
    """
    Decorator to deprecate a constructor argument and remove it from the signature.

    It works by popping the argument from kwargs, and setting it later via setattr.
    The property setter should take care of issuing the deprecation warning.

    Parameters
    ----------
    name : str
        Name of the argument to deprecate.

    Returns
    -------
    function
        decorated function
    """

    def wrapper(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            value = _UNSET
            if name in kwargs:
                value = kwargs.pop(name)
            res = func(*args, **kwargs)

            if value is not _UNSET:
                setattr(args[0], name, value)
            return res

        return _wrapper

    return wrapper

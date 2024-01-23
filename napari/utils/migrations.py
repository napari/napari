import inspect
import warnings
from functools import wraps

from napari.utils.translations import trans

_UNSET = object()


def rename_argument(
    from_name: str, to_name: str, version: str, since_version: str = ""
):
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
            "The since_version argument was added in napari 0.4.18 and will be mandatory since 0.6.0 release.",
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


def deprecated_constructor_arg_by_attr(name):
    """
    Decorator to deprecate a constructor argument and remove it from the signature.

    It works by popping the argument from kwargs, but thne setting it later via setattr.
    The property setter should take care of issuing the deprecation warning.

    Returns
    -------
    function
        decorated function
    """

    def wrapper(func):
        if not hasattr(func, '_deprecated_constructor_args'):
            func._deprecated_constructor_args = []
        func._deprecated_constructor_args.append(name)

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


def deprecated_class_name(
    new_class: type,
    previous_name: str,
    version: str,
    since_version: str,
) -> type:
    """Function to deprecate a class.

    Usage:

        class NewName:
            pass

        OldName = deprecated_class_name(
            NewName, 'OldName', version='0.5.0', since_version='0.4.19'
        )
    """
    msg = (
        f"{previous_name} is deprecated since {since_version} and will be "
        f"removed in {version}. Please use {new_class.__name__}."
    )
    prealloc_signature = inspect.signature(new_class.__new__)

    class _OldClass(new_class):
        def __new__(cls, *args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            if super().__new__ is object.__new__:
                return super().__new__(cls)
            return super().__new__(cls, *args, **kwargs)

        def __init_subclass__(cls, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)

    _OldClass.__module__ = new_class.__module__
    _OldClass.__name__ = previous_name
    _OldClass.__qualname__ = previous_name
    _OldClass.__new__.__signature__ = prealloc_signature  # type: ignore [attr-defined]

    return _OldClass

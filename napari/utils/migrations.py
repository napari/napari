import warnings
from functools import wraps
from typing import Any

from napari.utils.translations import trans


def rename_argument(from_name: str, to_name: str, version: str):
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
    """

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
                        "Argument {from_name} is deprecated, please use {to_name} instead. It will be removed in {version}.",
                        from_name=from_name,
                        to_name=to_name,
                        version=version,
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
) -> None:
    """
    Adds deprecated property and links to new property name setter and getter.

    Parameters
    ----------
    obj:
        Class instances to add property
    previous_name : str
        Name of previous property, it methods must be removed.
    new_name : str
        Name of new property, must have its setter and getter implemented.
    version : str
        Version where deprecated property will be removed.
    """

    if hasattr(obj, previous_name):
        raise RuntimeError(f"{previous_name} attribute already exists.")

    if not hasattr(obj, new_name):
        raise RuntimeError(f"{new_name} property must exists.")

    msg = trans._(
        f"{previous_name} is deprecated and will be removed in {version}. Please use {new_name}",
        deferred=True,
    )

    def _getter(instance) -> Any:
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
        return getattr(instance, new_name)

    def _setter(instance, value: Any) -> None:
        warnings.warn(msg, category=FutureWarning, stacklevel=3)
        setattr(instance, new_name, value)

    setattr(obj, previous_name, property(_getter, _setter))

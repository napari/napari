import warnings
from functools import wraps

from napari.utils.translations import trans


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

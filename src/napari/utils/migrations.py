"""Helpers for deprecating public API without breaking it.

See: https://napari.org/dev/developers/coredev/deprecation_policy.html
"""

import inspect
import warnings
from collections import UserDict
from functools import wraps
from typing import TYPE_CHECKING, Any, LiteralString, NamedTuple, overload

if TYPE_CHECKING:
    from collections.abc import Callable

_UNSET = object()

#: The canonical sentence. ``removal`` and ``replacement`` are themselves
#: filled in from the constants below so that the two states stay symmetrical.
DEPRECATION = (
    '{name} is deprecated{since_clause}.{removal}{replacement}{details}'
)

#: Filled into ``removal`` for a hard deprecation. ``window`` is ``YYYY-QN``
REMOVAL_SCHEDULED = ' It may be removed as early as {window}.'

#: Filled into ``removal`` for a soft deprecation.
REMOVAL_NOT_PLANNED = ' There are no current plans to remove it.'

#: Filled into ``replacement`` when there API to migrate to.
REPLACEMENT_INSTEAD = ' Please use {replacement} instead.'

#: Filled into ``replacement`` for a deprecation with no successor.
REPLACEMENT_NONE = ' There is no direct replacement.'


# ``typing_extensions.deprecated`` requires its message to be a ``LiteralString``
# When every argument is itself a literal the assembled sentence is a literal too
@overload
def deprecation_message(
    name: LiteralString,
    replacement: LiteralString = '',
    since: LiteralString = '',
    window: LiteralString | None = None,
    details: LiteralString = '',
) -> LiteralString: ...


@overload
def deprecation_message(
    name: str,
    replacement: str = '',
    since: str = '',
    window: str | None = None,
    details: str = '',
) -> str: ...


def deprecation_message(
    name: str,
    replacement: str = '',
    since: str = '',
    window: str | None = None,
    details: str = '',
) -> str:
    """Build a deprecation message in napari's canonical wording.

    Building every deprecation message here ensures that the wording is
    consistent and that ``is deprecated`` finds all of them.

    Parameters
    ----------
    name : str
        The deprecated name, for example ``'ViewerModel.camera'``.
    replacement : str
        What to use instead. Leave empty only when there is genuinely nothing
        to migrate to, which produces "There is no direct replacement."
    since : str
        The napari version the deprecation was introduced in.
    window : str, optional
        The removal window, as a ``YYYY-QN`` token meaning the name may be
        removed as early as that quarter. Omit it for a soft deprecation,
        which states that there are no current plans to remove the name.
    details : str
        Any further guidance that does not fit the canonical sentence, for
        example a migration recipe or a link to a guide.

    Returns
    -------
    str
        The full hard or soft deprecation message
    """
    return DEPRECATION.format(
        name=name,
        since_clause=f' since {since}' if since else '',
        removal=(
            REMOVAL_SCHEDULED.format(window=window)
            if window
            else REMOVAL_NOT_PLANNED
        ),
        replacement=(
            REPLACEMENT_INSTEAD.format(replacement=replacement)
            if replacement
            else REPLACEMENT_NONE
        ),
        details=f' {details}' if details else '',
    )


def deprecation_warning(
    name: str,
    replacement: str = '',
    *,
    since: str = '',
    window: str | None = None,
    details: str = '',
    stacklevel: int = 2,
) -> None:
    """Emit a canonical deprecation warning.

    The warning **category follows from ``window``**: a deprecation that names
    a removal window is hard and emits ``FutureWarning``, which end users see;
    one that does not is soft and emits ``DeprecationWarning``, which is more
    quiet and shows in IDEs (static type checkers), pytest output, and other
    development tools.

    Parameters
    ----------
    name, replacement, since, window, details
        Passed through to `deprecation_message`.
    stacklevel : int
        The stack level of the *caller*, defaulting to ``2``, meaning the
        warning points at whoever called this function.
    """
    warnings.warn(
        deprecation_message(
            name,
            replacement,
            since=since,
            window=window,
            details=details,
        ),
        category=_warning_category_for(window),
        stacklevel=stacklevel + 1,
    )


def _warning_category_for(window: str | None) -> type[Warning]:
    return FutureWarning if window else DeprecationWarning


class _RenamedAttribute(NamedTuple):
    """Captures information about a renamed attribute, property, or argument.

    Useful for storing internal state related to these types of deprecations.
    """

    from_name: str
    to_name: str
    window: str | None
    since_version: str

    def message(self) -> str:
        return deprecation_message(
            name=self.from_name,
            replacement=self.to_name,
            since=self.since_version,
            window=self.window,
        )


def rename_argument(
    from_name: str,
    to_name: str,
    window: str | None = None,
    since_version: str = '',
) -> 'Callable':
    """
    This is decorator for simple rename function argument
    without break backward compatibility.

    Parameters
    ----------
    from_name : str
        old name of argument
    to_name : str
        new name of argument
    window : str, optional
        Removal window as a ``YYYY-QN`` token, meaning the argument may be
        removed as early as that quarter. Omit it for a soft deprecation.
    since_version : str
        version when new argument was added
    """

    def _wrapper(func):
        if not hasattr(func, '_rename_argument'):
            func._rename_argument = []

        func._rename_argument.append(
            _RenamedAttribute(
                from_name=from_name,
                to_name=to_name,
                window=window,
                since_version=since_version,
            )
        )

        @wraps(func)
        def _update_from_dict(*args, **kwargs):
            if from_name in kwargs:
                if to_name in kwargs:
                    raise ValueError(
                        f'Argument {to_name} already defined, please do not mix {from_name} and {to_name} in one call.'
                    )
                warnings.warn(
                    deprecation_message(
                        name=f'Argument {from_name!r}',
                        replacement=f'{to_name!r}',
                        since=since_version,
                        window=window,
                    ),
                    category=_warning_category_for(window),
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
    window: str | None = None,
    since_version: str = '',
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
    window : str, optional
        Removal window as a ``YYYY-QN`` token, meaning the property may be
        removed as early as that quarter. Omit it for a soft deprecation.
    since_version : str
        version when new property was added
    """

    if hasattr(obj, previous_name):
        raise RuntimeError(f'{previous_name} property already exists.')

    if not hasattr(obj, new_name):
        raise RuntimeError(f'{new_name} property must exist.')

    msg = deprecation_message(
        name=f'{obj.__name__}.{previous_name}',
        replacement=new_name,
        since=since_version,
        window=window,
    )

    def _getter(instance) -> Any:
        warnings.warn(
            msg, category=_warning_category_for(window), stacklevel=3
        )
        return getattr(instance, new_name)

    def _setter(instance, value: Any) -> None:
        warnings.warn(
            msg, category=_warning_category_for(window), stacklevel=3
        )
        setattr(instance, new_name, value)

    setattr(obj, previous_name, property(_getter, _setter))


def deprecated_constructor_arg_by_attr(name: str) -> 'Callable':
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
    window: str | None = None,
    since_version: str = '',
) -> type:
    """Function to deprecate a class.

    Usage:

        class NewName:
            pass

        OldName = deprecated_class_name(
            NewName, 'OldName', window='2027-Q2', since_version='0.4.19'
        )
    """
    msg = deprecation_message(
        name=previous_name,
        replacement=new_class.__name__,
        since=since_version,
        window=window,
    )
    prealloc_signature = inspect.signature(new_class.__new__)

    class _OldClass(new_class):
        def __new__(cls, *args, **kwargs):
            warnings.warn(msg, _warning_category_for(window), stacklevel=2)
            if super().__new__ is object.__new__:
                return super().__new__(cls)
            return super().__new__(cls, *args, **kwargs)

        def __init_subclass__(cls, **kwargs):
            warnings.warn(msg, _warning_category_for(window), stacklevel=2)

    _OldClass.__module__ = new_class.__module__
    _OldClass.__name__ = previous_name
    _OldClass.__qualname__ = previous_name
    _OldClass.__new__.__signature__ = prealloc_signature  # pyrefly: ignore [missing-attribute]

    return _OldClass


class _DeprecatingDict(UserDict[str, Any]):
    """A dictionary that issues warning messages when deprecated keys are accessed.

    This class is intended to be an implementation detail of napari and may change
    in the future. As such, it should not be used outside of napari. Instead, treat
    this like a plain dictionary.

    Deprecated keys and values are not stored as part of the dictionary, so will not
    appear when iterating over this or its items.

    Instead deprecated items can only be accessed using `__getitem__`, `__setitem__`,
    and `__delitem__`.

    Deprecations from pure renames should keep the old and new corresponding items
    consistent when mutating either the old or new item.
    """

    # Maps from a deprecated key to its renamed key and deprecation information.
    _renamed: dict[str, _RenamedAttribute]

    def __init__(self, *args, **kwargs) -> None:
        self._renamed = {}
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: str) -> Any:
        key = self._maybe_rename_key(key)
        return self.data.__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:  # pyrefly: ignore [bad-override-param-name]
        key = self._maybe_rename_key(key)
        return self.data.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        key = self._maybe_rename_key(key)
        return self.data.__delitem__(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        key = self._maybe_rename_key(key)
        return self.data.__contains__(key)

    def _maybe_rename_key(self, key: str) -> str:
        if key in self._renamed:
            renamed = self._renamed[key]
            warnings.warn(
                renamed.message(), _warning_category_for(renamed.window)
            )
            key = renamed.to_name
        return key

    @property
    def deprecated_keys(self) -> tuple[str, ...]:
        return tuple(self._renamed.keys())

    def set_deprecated_from_rename(
        self,
        *,
        from_name: str,
        to_name: str,
        window: str | None = None,
        since_version: str = '',
    ) -> None:
        """Sets a deprecated key with a value that comes from another key.

        A warning message is automatically generated from the given window.
        """
        self._renamed[from_name] = _RenamedAttribute(
            from_name=from_name,
            to_name=to_name,
            window=window,
            since_version=since_version,
        )

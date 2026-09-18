from __future__ import annotations

import inspect
import warnings
from collections import UserDict
from functools import wraps
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

_UNSET = object()


class RenamedProperty(property):
    """A deprecated alias that forwards access to a renamed or moved attribute.

    Reading or writing the alias emits a warning and accesses the target
    attribute. Access through the class returns the descriptor without warning.

    Parameters
    ----------
    new_name : str
        Target attribute name or dotted path relative to the instance, such as
        ``appearance.size``. Each path component must be a Python identifier.
    since_version : str
        Version in which the alias was deprecated. An empty string omits the
        version from warning messages.
    due_date : str, optional
        Announced removal date or season, such as ``spring 2027``. If omitted,
        no removal date is announced. This does not automatically disable access.
    category : type of Warning, optional
        Warning category used for access and assignment. Defaults to FutureWarning.
    writable : bool, optional
        Whether assignment through the alias is allowed. Defaults to True.
        The target must also support assignment.
    doc : str, optional
        Documentation for the alias. A deprecation directive is appended unless
        the text already contains ``.. deprecated::``.

    Raises
    ------
    ValueError
        If the target path is empty or contains an invalid component.

    Notes
    -----
    Declare the descriptor in the class body to initialize its owner and name
    automatically. When attaching it after class creation, call ``__set_name__``
    explicitly. Define custom accessors on the target property; the inherited
    ``getter``, ``setter``, and ``deleter`` helpers are not supported.

    Examples
    --------
    >>> class Settings:
    ...     size = 10
    ...     old_size = RenamedProperty(new_name='size', since_version='0.7.0')
    """

    def __init__(
        self,
        *,
        new_name: str,
        since_version: str,
        due_date: str | None = None,
        category: type[Warning] = FutureWarning,
        writable: bool = True,
        doc: str | None = None,
    ):
        parts = new_name.split('.')
        if any(not part.isidentifier() for part in parts):  # pragma: no cover
            raise ValueError(f'Invalid attribute path: {new_name!r}')

        self._new_name = new_name
        self._since_version = since_version
        self._due_date = due_date
        self._category = category
        self._owner_name: str | None = None
        self._name: str | None = None
        self._parent_path = tuple(parts[:-1])
        self._target_name = parts[-1]
        doc = doc or ''
        if '.. deprecated::' not in doc:
            doc += (
                f'\n\n.. deprecated:: {since_version}\n'
                f'    Use `{new_name}` instead.\n'
            )
        super().__init__(
            fget=self._get_value,
            fset=self._set_value if writable else None,
            doc=doc,
        )

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the owning class and alias name for warning messages."""
        self._owner_name = owner.__qualname__
        self._name = name

    @property
    def new_name(self) -> str:
        """Target attribute name or dotted path relative to the instance."""
        return self._new_name

    @property
    def category(self) -> type[Warning]:
        """Warning category used when reading or writing the alias."""
        return self._category

    @property
    def name(self) -> str:
        """Alias name."""
        if self._name is None:  # pragma: no cover
            raise RuntimeError(
                'RenamedProperty has not been assigned to a class yet.'
            )
        return self._name

    @property
    def message(self) -> str:
        """Deprecation warning text for access to the aliased attribute."""
        name = (
            f'{self._owner_name}.{self._name}'
            if self._name is not None
            else 'This property'
        )
        since = f' since {self._since_version}' if self._since_version else ''
        schedule = (
            f' Removal is scheduled for {self._due_date}.'
            if self._due_date is not None
            else ''
        )
        return (
            f'{name} is deprecated{since}.'
            f'{schedule} Please use {self._new_name} instead.'
        )

    @property
    def event_message(self) -> str:
        """Deprecation warning text for the corresponding renamed event.

        The replacement event belongs to the target attribute's parent object:
        a target of ``appearance.size`` uses ``appearance.events.size``.
        This property only provides the message; it does not create an emitter.
        """
        name = (
            f'{self._owner_name}.events.{self._name}'
            if self._name is not None
            else 'This event'
        )
        since = f' since {self._since_version}' if self._since_version else ''
        schedule = (
            f' Removal is scheduled for {self._due_date}.'
            if self._due_date is not None
            else ''
        )
        new_name = f'{self._owner_name}.{".".join(self._parent_path + ("events", self._target_name))}'
        return (
            f'{name} is deprecated{since}.'
            f'{schedule} Please use {new_name} instead.'
        )

    def _get_value(self, instance: object) -> object:
        """Warn and read the target attribute."""
        warnings.warn(self.message, self._category, stacklevel=2)
        return getattr(self._resolve_parent(instance), self._target_name)

    def _set_value(self, instance: object, value: object) -> None:
        """Warn and assign to the target attribute."""
        warnings.warn(self.message, self._category, stacklevel=2)
        setattr(self._resolve_parent(instance), self._target_name, value)

    def _resolve_parent(self, instance: object) -> object:
        """Resolve the object that holds the final attribute in the target path."""
        target = instance
        for part in self._parent_path:
            target = getattr(target, part)
        return target


class _RenamedAttribute(NamedTuple):
    """Captures information about a renamed attribute, property, or argument.

    Useful for storing internal state related to these types of deprecations.
    """

    from_name: str
    to_name: str
    version: str
    since_version: str

    def message(self) -> str:
        return f'{self.from_name} is deprecated since {self.since_version} and will be removed in {self.version}. Please use {self.to_name}'


def rename_argument(
    from_name: str, to_name: str, version: str, since_version: str = ''
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

    def _wrapper(func):
        if not hasattr(func, '_rename_argument'):
            func._rename_argument = []

        func._rename_argument.append(
            _RenamedAttribute(
                from_name=from_name,
                to_name=to_name,
                version=version,
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
                    f'Argument {from_name!r} is deprecated, please use {to_name!r} instead. The argument {from_name!r} was deprecated in {since_version} and it will be removed in {version}.',
                    category=FutureWarning,
                    stacklevel=2,
                )
                kwargs = kwargs.copy()
                kwargs[to_name] = kwargs.pop(from_name)
            return func(*args, **kwargs)

        return _update_from_dict

    return _wrapper


def _add_deprecated_property(func):
    """To be used as a decorator for add_deprecated_property to support legacy positional arguments."""

    @wraps(func)
    def _func(*args, **kwargs):
        if args or 'obj' in kwargs:
            if args:
                warnings.warn(
                    'Using positional arguments for add_deprecated_property is deprecated. '
                    'Please use keyword arguments instead. '
                    'positional arguments will be removed in a spring 2027',
                    category=FutureWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Using 'obj' keyword argument for add_deprecated_property is deprecated. "
                    'Please use add_deprecated_property(...)(obj) instead. ',
                    category=FutureWarning,
                    stacklevel=2,
                )
            if 'obj' in kwargs:
                obj = kwargs.pop('obj')
            else:
                obj = args[0]
                args = args[1:]

            legacy_names = (
                'previous_name',
                'new_name',
                'version',
                'since_version',
            )
            for name, value in zip(legacy_names, args, strict=False):
                kwargs[name] = value
            return func(**kwargs)(obj)
        return func(**kwargs)

    return _func


@_add_deprecated_property
def add_deprecated_property(
    *,
    previous_name: str,
    new_name: str,
    version: str | None = None,
    since_version: str = '',
    due_date: str | None = None,
) -> Callable[[type], type]:
    """
    Adds deprecated property and links to new property name setter and getter.

    Parameters
    ----------
    previous_name : str
        Name of previous property, its methods must be removed.
    new_name : str
        Name of new property, must have its getter (and setter if applicable) implemented.
    version : str, optional
        Deprecated and ignored. Use ``due_date`` to announce a removal date.
    since_version : str
        version when new property was added
    due_date : str, optional
        Announced removal date or season, such as "spring 2027". If omitted,
        no removal date is announced. This does not automatically disable access.

    ..deprecated:: 0.9.2
        `version` argument is deprecated and ignored. Use `due_date` to specify a removal date, or omit it for no announced removal date.
    """
    if version is not None:
        warnings.warn(
            "The 'version' argument to add_deprecated_property is deprecated "
            "and ignored. Use 'due_date' to specify a removal date, or omit "
            'it for no announced removal date.',
            FutureWarning,
            stacklevel=3,
        )

    def _func(obj: type) -> type:
        if hasattr(obj, previous_name):
            raise RuntimeError(f'{previous_name} property already exists.')

        if not hasattr(obj, new_name):
            raise RuntimeError(f'{new_name} property must exist.')

        prop = RenamedProperty(
            new_name=new_name, since_version=since_version, due_date=due_date
        )
        setattr(obj, previous_name, prop)
        prop.__set_name__(obj, previous_name)
        return obj

    return _func


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
        f'{previous_name} is deprecated since {since_version} and will be '
        f'removed in {version}. Please use {new_class.__name__}.'
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

    def __setitem__(self, key: str, value: Any) -> None:
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
            warnings.warn(renamed.message(), FutureWarning)
            key = renamed.to_name
        return key

    @property
    def deprecated_keys(self) -> tuple[str, ...]:
        return tuple(self._renamed.keys())

    def set_deprecated_from_rename(
        self, *, from_name: str, to_name: str, version: str, since_version: str
    ) -> None:
        """Sets a deprecated key with a value that comes from another key.

        A warning message is automatically generated using the given version information.
        """
        self._renamed[from_name] = _RenamedAttribute(
            from_name=from_name,
            to_name=to_name,
            version=version,
            since_version=since_version,
        )

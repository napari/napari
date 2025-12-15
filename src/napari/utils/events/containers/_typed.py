from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)

from napari.utils.translations import trans

logger = logging.getLogger(__name__)


Index = int | slice

_T = TypeVar('_T')
_L = TypeVar('_L', bound=Any)
_K = TypeVar('_K')


class TypedLookupSequenceMixin(Generic[_T]):
    """Sequence mixin that enforces item type and provides lookup functionality."""

    def __init__(
        self,
        *args,
        basetype: type[_T] | Sequence[type[_T]] = (),
        lookup: dict[type[_L], Callable[[_T], _T | _L]] | None = None,
        **kwargs,
    ) -> None:
        if lookup is None:
            lookup = {}
        self._basetypes: tuple[type[_T], ...] = (
            tuple(basetype) if isinstance(basetype, Sequence) else (basetype,)
        )
        self._lookup = lookup.copy()
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: Index, value: _T):
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError(
                    trans._(
                        'Can only assign an iterable to slice',
                        deferred=True,
                    )
                )
            super().__setitem__(key, [self._type_check(v) for v in value])
        else:
            super().__setitem__(key, self._type_check(value))

    def insert(self, index: int, value: _T) -> None:
        super().insert(index, self._type_check(value))

    def __contains__(self, key: Any) -> bool:
        if type(key) in self._lookup:
            try:
                self[self.index(key)]
            except ValueError:
                return False
            else:
                return True
        return super().__contains__(key)

    @overload
    def __getitem__(self, key: str) -> _T: ...  # pragma: no cover

    @overload
    def __getitem__(self, key: int) -> _T: ...  # pragma: no cover

    @overload
    def __getitem__(
        self, key: slice
    ) -> TypedLookupSequenceMixin[_T]: ...  # pragma: no cover

    def __getitem__(self, key):
        """Get an item from the list

        Parameters
        ----------
        key : int, slice, or any type in self._lookup
            The key to get.

        Returns
        -------
        The value at `key`

        Raises
        ------
        IndexError:
            If ``type(key)`` is not in ``self._lookup`` (usually an int, like a regular
            list), and the index is out of range.
        KeyError:
            If type(key) is in self._lookup and the key is not in the list (after)
            applying the self._lookup[key] function to each item in the list
        """
        if type(key) in self._lookup:
            try:
                return self[self.index(key)]
            except ValueError as e:
                raise KeyError(str(e)) from e

        return self[key]

    def __delitem__(self, key) -> None:
        _key = self.index(key) if type(key) in self._lookup else key
        del self[_key]

    def _type_check(self, e: Any) -> _T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                trans._(
                    'Cannot add object with type {dtype!r} to TypedList expecting type {basetypes!r}',
                    deferred=True,
                    dtype=type(e),
                    basetypes=self._basetypes,
                )
            )
        return e

    def __newlike__(
        self, iterable: Iterable[_T]
    ) -> TypedLookupSequenceMixin[_T]:
        new = self.__class__()
        # separating this allows subclasses to omit these from their `__init__`
        new._basetypes = self._basetypes
        new._lookup = self._lookup.copy()
        new.extend(iterable)
        return new

    def index(self, value: _L, start: int = 0, stop: int | None = None) -> int:
        """Return first index of value.

        Parameters
        ----------
        value : Any
            A value to lookup.  If `type(value)` is in the lookups functions
            provided for this class, then values in the list will be searched
            using the corresponding lookup converter function.
        start : int, optional
            The starting index to search, by default 0
        stop : int, optional
            The ending index to search, by default None

        Returns
        -------
        int
            The index of the value

        Raises
        ------
        ValueError
            If the value is not present
        """
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        convert = self._lookup.get(type(value), lambda x: x)

        for i in self._iter_indices(start, stop):
            v = convert(self[i])
            if v is value or v == value:
                return i

        raise ValueError(
            trans._(
                '{value!r} is not in list',
                deferred=True,
                value=value,
            )
        )

    def _iter_indices(
        self, start: int = 0, stop: int | None = None
    ) -> Iterable[int]:
        """Iter indices from start to stop.

        While this is trivial for this basic sequence type, this method lets
        subclasses (like NestableEventedList modify how they are traversed).
        """
        yield from range(start, len(self) if stop is None else stop)

    def _ipython_key_completions_(self):
        if str in self._lookup:
            return (self._lookup[str](x) for x in self)
        return None  # type: ignore


class TypedMappingMixin(Generic[_K, _T]):
    """Dictionary mixin that enforces item type."""

    def __init__(
        self,
        *args,
        basetype: type[_T] | Sequence[type[_T]] = (),
        **kwargs,
    ) -> None:
        self._dict: dict[_K, _T] = {}
        self._basetypes = (
            tuple(basetype) if isinstance(basetype, Sequence) else (basetype,)
        )
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: _K, value: _T) -> None:
        super().__setitem__(key, self._type_check(value))

    def _type_check(self, e: Any) -> _T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                f'Cannot add object with type {type(e)} to TypedDict expecting type {self._basetypes}',
            )
        return e

    def __newlike__(
        self, iterable: TypedMappingMixin[_K, _T]
    ) -> TypedMappingMixin[_K, _T]:
        new = self.__class__()
        # separating this allows subclasses to omit these from their `__init__`
        new._basetypes = self._basetypes
        new.update(iterable)
        return new

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableSequence,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from ._list import EventedList, T
from ._nested_list import NestableEventedList

L = TypeVar("L")


# the typing here is a little loose... This mixin MUST be used to extend a
# mutable sequence, but I struggled getting a consistent MRO when I tried to
# type it the "right" way.
class TypedMixin:
    """List mixin that enforces item type, and enables custom indexing.

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the list with.
    basetype : type
        Type of the elements in the list.
    lookup : dict of Type[L] : function(object) -> L
        Mapping between a type, and a function that converts items in the list
        to that type.
    """

    def __init__(
        self,
        data: Iterable[T] = (),
        basetype: Union[Type[T], Sequence[Type[T]]] = (),
        lookup: Dict[Type[L], Callable[[T], L]] = None,
    ):
        self._basetypes = (
            basetype if isinstance(basetype, Sequence) else (basetype,)
        )
        self._lookup = lookup or {}
        super().__init__(data)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError('Can only assign an iterable to slice')
            _value = [self._type_check(v) for v in value]
        else:
            _value = self._type_check(value)
        super().__setitem__(key, _value)

    def insert(self, index: int, value: T):
        super().insert(index, self._type_check(value))

    def __contains__(self, key):
        if type(key) in self._lookup:
            try:
                self[self.index(key)]
            except ValueError:
                return False
            else:
                return True
        return super().__contains__(key)

    def __getitem__(self, key):
        if type(key) in self._lookup:
            return super().__getitem__(self.index(key))
        return super().__getitem__(key)

    def __delitem__(self, key):
        if type(key) in self._lookup:
            return super().__delitem__(self.index(key))
        return super().__delitem__(key)

    def _type_check(self, e: Any) -> T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                f'Cannot add object with type {type(e)!r} to '
                f'TypedList expecting type {self._basetypes!r}'
            )
        return e

    def __newlike__(self, iterable: Iterable[T]):
        return self.__class__(iterable, self._basetypes, self._lookup)

    def index(self, value: Any, start: int = 0, stop: int = None) -> int:
        """Return first index of value.

        Parameters
        ----------
        value : Any
            A value to lookup
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

        i = start
        while stop is None or i < stop:
            try:
                v = convert(self[i])
                if v is value or v == value:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError(f"{value!r} is not in list")

    def _ipython_key_completions_(self):
        if str in self._lookup:
            return (self._lookup[str](x) for x in self)


class ConcreteMutableSequence(MutableSequence[T]):
    def __init__(
        self, data: Iterable[T] = None,
    ):
        self._list: List[T] = []
        self.extend(data or [])

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return repr(self._list)

    def __setitem__(self, key, value):
        self._list[key] = value

    def insert(self, index: int, value: T):
        self._list.insert(index, value)

    def __getitem__(self, key):
        return self._list[key]

    def __delitem__(self, key):
        del self._list[key]


class TypedList(TypedMixin, ConcreteMutableSequence):
    pass


class TypedEventedList(TypedMixin, EventedList):
    pass


class TypedNestableEventedList(TypedMixin, NestableEventedList):
    def _type_check(self, e: Any) -> T:
        if isinstance(e, list):
            return self.__newlike__(e)
        if self._basetypes:
            _types = list(self._basetypes) + [TypedNestableEventedList]
            if not any(isinstance(e, t) for t in _types):
                raise TypeError(
                    f'Cannot add object with type {type(e)!r} to '
                    f'TypedList expecting type {_types!r}'
                )
        return e

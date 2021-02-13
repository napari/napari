import logging
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
    overload,
)

logger = logging.getLogger(__name__)


Index = Union[int, slice]

_T = TypeVar("_T")
_L = TypeVar("_L")


class B(MutableSequence[int]):
    def __new__(self, data):
        self._data = data


class TypedMutableSequence(MutableSequence[_T]):
    """List mixin that enforces item type, and enables custom indexing.

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the list with.
    basetype : type or sequence of types, optional
        Type of the elements in the list.  If a basetype (or multiple) is
        provided, then a TypeError will be raised when attempting to add an
        item to this sequence if it is not an instance of one of the types in
        ``basetype``.
    lookup : dict of Type[L] : function(object) -> L
        Mapping between a type, and a function that converts items in the list
        to that type.  This is used for custom indexing.  For example, if a
        ``lookup`` of {str: lambda x: x.name} is provided, then you can index
        into the list using ``list['frank']`` and it will search for an object
        whos attribute ``.name`` equals ``'frank'``.
    """

    # required for inspect.sigature to be correct...
    def __new__(
        cls,
        data=(),
        *,
        basetype=(),
        lookup=dict(),
    ):
        return object.__new__(cls)

    def __init__(
        self,
        data: Iterable[_T] = (),
        *,
        basetype: Union[Type[_T], Sequence[Type[_T]]] = (),
        lookup: Dict[Type[_L], Callable[[_T], Union[_T, _L]]] = dict(),
    ):
        self._list: List[_T] = []
        self._basetypes = (
            basetype if isinstance(basetype, Sequence) else (basetype,)
        )
        self._lookup = lookup.copy()
        self.extend(data)

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return repr(self._list)

    def __eq__(self, other: Any):
        return self._list == other

    def __hash__(self) -> int:
        # it's important to add this to allow this object to be hashable
        # given that we've also reimplemented __eq__
        return id(self)

    @overload
    def __setitem__(self, key: int, value: _T):  # noqa: F811
        ...  # pragma: no cover

    @overload
    def __setitem__(self, key: slice, value: Iterable[_T]):  # noqa: F811
        ...  # pragma: no cover

    def __setitem__(self, key, value):  # noqa: F811
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError('Can only assign an iterable to slice')
            self._list[key] = [self._type_check(v) for v in value]
        else:
            self._list[key] = self._type_check(value)

    def insert(self, index: int, value: _T):
        self._list.insert(index, self._type_check(value))

    def __contains__(self, key):
        if type(key) in self._lookup:
            try:
                self[self.index(key)]
            except ValueError:
                return False
            else:
                return True
        return super().__contains__(key)

    @overload
    def __getitem__(self, key: int) -> _T:  # noqa: F811
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: slice) -> 'TypedMutableSequence[_T]':  # noqa
        ...  # pragma: no cover

    def __getitem__(self, key):  # noqa: F811
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
                _key = self.index(key)
            except ValueError as e:
                raise KeyError(str(e)) from e
        else:
            _key = key
        result = self._list[_key]
        return self.__newlike__(result) if isinstance(result, list) else result

    def __delitem__(self, key):
        _key = self.index(key) if type(key) in self._lookup else key
        del self._list[_key]

    def _type_check(self, e: Any) -> _T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                f'Cannot add object with type {type(e)!r} to '
                f'TypedList expecting type {self._basetypes!r}'
            )
        return e

    def __newlike__(self, iterable: Iterable[_T]):
        return self.__class__(
            iterable, basetype=self._basetypes, lookup=self._lookup
        )

    def copy(self) -> 'TypedMutableSequence[_T]':
        """Return a shallow copy of the list."""
        return self.__newlike__(self)

    def __add__(self, other: Iterable[_T]) -> 'TypedMutableSequence[_T]':
        """Add other to self, return new object."""
        copy = self.copy()
        copy.extend(other)
        return copy

    def __iadd__(self, other: Iterable[_T]) -> 'TypedMutableSequence[_T]':
        """Add other to self in place (self += other)."""
        self.extend(other)
        return self

    def __radd__(self, other: List) -> List:
        """Add other to self in place (self += other)."""
        return other + list(self)

    def index(self, value: _L, start: int = 0, stop: int = None) -> int:
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

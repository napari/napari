"""Evented dictionary"""
import sys
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    TypeVar,
    Union,
)

_K = TypeVar("_K")
_T = TypeVar("_T")


class TypedMutableMapping(MutableMapping[_K, _T]):
    """Dictionary mixin that enforces item type."""

    def __init__(
        self,
        data: Mapping[_K, _T] = None,
        basetype: Union[Type[_T], Sequence[Type[_T]]] = (),
    ):
        if data is None:
            data = {}
        self._dict: Dict[_K, _T] = dict()
        self._basetypes = (
            basetype if isinstance(basetype, Sequence) else (basetype,)
        )
        self.update(data)

    # #### START Required Abstract Methods

    def __setitem__(self, key: int, value: _T):  # noqa: F811
        self._dict[key] = self._type_check(value)

    def __delitem__(self, key: _K) -> None:
        del self._dict[key]

    def __getitem__(self, key: _K) -> _T:
        return self._dict[key]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[_T]:
        return iter(self._dict)

    def __repr__(self):
        return str(self._dict)

    if sys.version_info < (3, 8):

        def __hash__(self):
            # We've explicitly added __hash__ for python < 3.8 because otherwise
            # nested evented dictionaries fail tests.
            # This can be removed once we drop support for python < 3.8
            # see: https://github.com/napari/napari/pull/2994#issuecomment-877105434
            return hash(frozenset(self))

    def _type_check(self, e: Any) -> _T:
        if self._basetypes and not any(
            isinstance(e, t) for t in self._basetypes
        ):
            raise TypeError(
                f"Cannot add object with type {type(e)} to TypedDict expecting type {self._basetypes}",
            )
        return e

    def __newlike__(self, iterable: MutableMapping[_K, _T]):
        new = self.__class__()
        # separating this allows subclasses to omit these from their `__init__`
        new._basetypes = self._basetypes
        new.update(**iterable)
        return new

    def copy(self) -> "TypedMutableMapping[_T]":
        """Return a shallow copy of the dictionary."""
        return self.__newlike__(self)

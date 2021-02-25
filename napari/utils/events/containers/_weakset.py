from abc import ABCMeta
from typing import Any, Iterable, TypeVar, Union
from weakref import ReferenceType, ref

from ._set import EventedSet

_T = TypeVar("_T")


class _WeakMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        print(bases, dct)
        return x


class T(EventedSet[_T], metaclass=_WeakMeta):
    def t(se):
        ...


class EventedWeakSet(EventedSet['ReferenceType[_T]'], metaclass=_WeakMeta):
    """An EventedSet variant that only stores weakrefs."""

    def add(self, value: Union[_T, 'ReferenceType[_T]']) -> None:
        vref = self._ensure_ref(value)
        if vref not in self:
            self._set.add(vref)
            self.events.added(value={value})

    def discard(self, value: Union[_T, 'ReferenceType[_T]']) -> None:
        vref = self._ensure_ref(value)
        if vref in self:
            self._set.discard(vref)
            self.events.removed(value={value})

    def update(
        self, others: Iterable[Union[_T, 'ReferenceType[_T]']] = ()
    ) -> None:
        """Update this set with the union of this set and others"""
        orefs = {self._ensure_ref(o) for o in others}
        to_add = orefs.difference(self._set)
        if to_add:
            self._set.update(to_add)
            self.events.added(value={i() for i in to_add})

    @staticmethod
    def _ensure_ref(val):
        return ref(val) if not isinstance(val, ReferenceType) else val

    def __contains__(self, x: Any) -> bool:
        return super().__contains__(self._ensure_ref(x))

    def __iter__(self):
        for x in iter(self._set):
            yield x()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(set(self))})"

    for n in (
        'difference',
        'difference_update',
        'intersection',
        'intersection_update',
        'issubset',
        'issuperset',
        'symmetric_difference',
        'symmetric_difference_update',
    ):
        setattr()

    # def difference(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
    #     """Return set of all elements that are in this set but not other."""
    #     return type(self)(self._set.difference(others))

    # def difference_update(self, others: Iterable[_T] = ()) -> None:
    #     """Remove all elements of another set from this set."""
    #     to_remove = self._set.intersection(others)
    #     if to_remove:
    #         self._set.difference_update(to_remove)
    #         self.events.removed(value=to_remove)

    # def intersection(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
    #     """Return all elements that are in both sets as a new set."""
    #     return type(self)(self._set.intersection(others))

    # def intersection_update(self, others: Iterable[_T] = ()) -> None:
    #     """Remove all elements of in this set that are not present in other."""
    #     self.difference_update(self._set.symmetric_difference(others))

    # def issubset(self, others: Iterable[_T]) -> bool:
    #     """Returns whether another set contains this set or not"""
    #     return self._set.issubset(others)

    # def issuperset(self, others: Iterable[_T]) -> bool:
    #     """Returns whether this set contains another set or not"""
    #     return self._set.issuperset(others)

    # def symmetric_difference(self, others: Iterable[_T]) -> EventedSet[_T]:
    #     """Returns set of elements that are in exactly one of the sets"""
    #     return type(self)(self._set.symmetric_difference(others))

    # def symmetric_difference_update(self, others: Iterable[_T]) -> None:
    #     """Update set to the symmetric difference of itself and another.

    #     This will remove any items in this set that are also in `other`, and
    #     add any items in others that are not present in this set.
    #     """
    #     to_add = set(others).difference(self)
    #     self.difference_update(others)
    #     self.update(to_add)

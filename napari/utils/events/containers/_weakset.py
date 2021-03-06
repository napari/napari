from typing import Any, Iterable, TypeVar, Union
from weakref import ReferenceType, ref

from napari.utils.events.containers._set import EventedSet

_T = TypeVar("_T")


def _ensure_ref(val):
    return ref(val) if not isinstance(val, ReferenceType) else val


class EventedWeakSet(EventedSet['ReferenceType[_T]']):
    """An EventedSet variant that only stores weakrefs."""

    def add(self, value: Union[_T, 'ReferenceType[_T]']) -> None:
        vref = _ensure_ref(value)
        if vref not in self:
            self._set.add(vref)
            self.events.added(value={value})

    def discard(self, value: Union[_T, 'ReferenceType[_T]']) -> None:
        vref = _ensure_ref(value)
        if vref in self:
            self._set.discard(vref)
            self.events.removed(value={value})

    def update(
        self, others: Iterable[Union[_T, 'ReferenceType[_T]']] = ()
    ) -> None:
        """Update this set with the union of this set and others"""
        orefs = {_ensure_ref(o) for o in others}
        to_add = orefs.difference(self._set)
        if to_add:
            self._set.update(to_add)
            self.events.added(value={i() for i in to_add})

    def __contains__(self, x: Any) -> bool:
        return super().__contains__(_ensure_ref(x))

    def __iter__(self):
        for x in iter(self._set):
            yield x()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(set(self))})"

    def difference(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        raise NotImplementedError("not all set methods available on weakset")

    def difference_update(self, others: Iterable[_T] = ()) -> None:
        raise NotImplementedError("not all set methods available on weakset")

    def intersection(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        raise NotImplementedError("not all set methods available on weakset")

    def intersection_update(self, others: Iterable[_T] = ()) -> None:
        raise NotImplementedError("not all set methods available on weakset")

    def issubset(self, others: Iterable[_T]) -> bool:
        raise NotImplementedError("not all set methods available on weakset")

    def issuperset(self, others: Iterable[_T]) -> bool:
        raise NotImplementedError("not all set methods available on weakset")

    def symmetric_difference(self, others: Iterable[_T]) -> EventedSet[_T]:
        raise NotImplementedError("not all set methods available on weakset")

    def symmetric_difference_update(self, others: Iterable[_T]) -> None:
        raise NotImplementedError("not all set methods available on weakset")

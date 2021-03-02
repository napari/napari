from __future__ import annotations

from typing import Any, Iterable, Iterator, MutableSet, TypeVar

from napari.utils.events import EmitterGroup

_T = TypeVar("_T")


class EventedSet(MutableSet[_T]):
    """An unordered collection of unique elements.

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the set with.

    Events
    ------
    added (value: Set[_T])
        emitted after an item or items are added to the set.
        Will not be emitted if item was already in the set when added.
    removed (value: Set[_T])
        emitted after an item or items are removed from the set.
        Will not be emitted if the item was not in the set when discarded.
    """

    events: EmitterGroup

    def __init__(self, data: Iterable[_T] = ()):

        _events = {'added': None, 'removed': None}
        # For inheritance: If the mro already provides an EmitterGroup, add...
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            # otherwise create a new one
            self.events = EmitterGroup(source=self, **_events)

        self._set: set[_T] = set()
        self.update(data)

    # #### START Required Abstract Methods

    def __contains__(self, x: Any) -> bool:
        return x in self._set

    def __iter__(self) -> Iterator[_T]:
        return iter(self._set)

    def __len__(self) -> int:
        return len(self._set)

    def add(self, value: _T) -> None:
        if value not in self:
            self._set.add(value)
            self.events.added(value={value})

    def discard(self, value: _T) -> None:
        """Remove an element from a set if it is a member.

        If the element is not a member, do nothing.
        """
        if value in self:
            self._set.discard(value)
            self.events.removed(value={value})

    # #### END Required Abstract Methods

    # methods inherited from Set:
    # __le__, __lt__, __eq__, __ne__, __gt__, __ge__, __and__, __or__,
    # __sub__, __xor__, and isdisjoint

    # methods inherited from MutableSet:
    # clear, pop, remove, __ior__, __iand__, __ixor__, and __isub__

    # The rest are for parity with builtins.set:

    def clear(self) -> None:
        if self._set:
            values = set(self._set)
            self._set.clear()
            self.events.removed(value=values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._set)})"

    def update(self, others: Iterable[_T] = ()) -> None:
        """Update this set with the union of this set and others"""
        to_add = set(others).difference(self._set)
        if to_add:
            self._set.update(to_add)
            self.events.added(value=to_add)

    def copy(self) -> EventedSet[_T]:
        """Return a shallow copy of this set."""
        return EventedSet(self._set)

    def difference(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        """Return set of all elements that are in this set but not other."""
        return EventedSet(self._set.difference(others))

    def difference_update(self, others: Iterable[_T] = ()) -> None:
        """Remove all elements of another set from this set."""
        to_remove = self._set.intersection(others)
        if to_remove:
            self._set.difference_update(others)
            self.events.removed(value=to_remove)

    def intersection(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        """Return all elements that are in both sets as a new set."""
        return EventedSet(self._set.intersection(others))

    def intersection_update(self, others: Iterable[_T] = ()) -> None:
        """Remove all elements of in this set that are not present in other."""
        self.difference_update(self._set.symmetric_difference(others))

    def issubset(self, others: Iterable[_T]) -> bool:
        """Returns whether another set contains this set or not"""
        return self._set.issubset(others)

    def issuperset(self, others: Iterable[_T]) -> bool:
        """Returns whether this set contains another set or not"""
        return self._set.issuperset(others)

    def symmetric_difference(self, others: Iterable[_T]) -> EventedSet[_T]:
        """Returns set of elements that are in exactly one of the sets"""
        return EventedSet(self._set.symmetric_difference(others))

    def symmetric_difference_update(self, others: Iterable[_T]) -> None:
        """Update set to the symmetric difference of itself and another.

        This will remove any items in this set that are also in `other`, and
        add any items in others that are not present in this set.
        """
        to_remove = self._set.intersection(others)
        to_add = set(others).difference(self)
        if to_remove:
            self._set.difference_update(others)
            self.events.removed(value=to_remove)
        if to_add:
            self._set.update(to_add)
            self.events.added(value=to_add)

    def union(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        """Return a set containing the union of sets"""
        return EventedSet(self._set.union(others))

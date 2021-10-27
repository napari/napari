from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, MutableSet, TypeVar

from napari.utils.events import EmitterGroup

from ....utils.translations import trans

_T = TypeVar("_T")

if TYPE_CHECKING:
    from pydantic.fields import ModelField


class EventedSet(MutableSet[_T]):
    """An unordered collection of unique elements.

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the set with.

    Events
    ------
    changed (added: Set[_T], removed: Set[_T])
        Emitted when the set changes, includes item(s) that have been added
        and/or removed from the set.
    """

    events: EmitterGroup

    def __init__(self, data: Iterable[_T] = ()):

        _events = {'changed': None}
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

    def _pre_add_hook(self, value):
        # for subclasses to potentially check value before adding
        return value

    def add(self, value: _T) -> None:
        """Add an element to the set, if not already present."""
        if value not in self:
            value = self._pre_add_hook(value)
            self._set.add(value)
            self.events.changed(added={value}, removed={})

    def discard(self, value: _T) -> None:
        """Remove an element from a set if it is a member.

        If the element is not a member, do nothing.
        """
        if value in self:
            self._set.discard(value)
            self.events.changed(added={}, removed={value})

    # #### END Required Abstract Methods

    # methods inherited from Set:
    # __le__, __lt__, __eq__, __ne__, __gt__, __ge__, __and__, __or__,
    # __sub__, __xor__, and isdisjoint

    # methods inherited from MutableSet:
    # clear, pop, remove, __ior__, __iand__, __ixor__, and __isub__

    # The rest are for parity with builtins.set:

    def clear(self) -> None:
        if self._set:
            values = set(self)
            self._set.clear()
            self.events.changed(added={}, removed=values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._set)})"

    def update(self, others: Iterable[_T] = ()) -> None:
        """Update this set with the union of this set and others"""
        to_add = set(others).difference(self._set)
        if to_add:
            to_add = {self._pre_add_hook(i) for i in to_add}
            self._set.update(to_add)
            self.events.changed(added=set(to_add), removed={})

    def copy(self) -> EventedSet[_T]:
        """Return a shallow copy of this set."""
        return type(self)(self._set)

    def difference(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        """Return set of all elements that are in this set but not other."""
        return type(self)(self._set.difference(others))

    def difference_update(self, others: Iterable[_T] = ()) -> None:
        """Remove all elements of another set from this set."""
        to_remove = self._set.intersection(others)
        if to_remove:
            self._set.difference_update(to_remove)
            self.events.changed(added={}, removed=set(to_remove))

    def intersection(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        """Return all elements that are in both sets as a new set."""
        return type(self)(self._set.intersection(others))

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
        return type(self)(self._set.symmetric_difference(others))

    def symmetric_difference_update(self, others: Iterable[_T]) -> None:
        """Update set to the symmetric difference of itself and another.

        This will remove any items in this set that are also in `other`, and
        add any items in others that are not present in this set.
        """
        to_add = set(others).difference(self._set)
        to_remove = self._set.intersection(others)
        self._set.difference_update(to_remove)
        self._set.update(to_add)
        self.events.changed(added=to_add, removed=to_remove)

    def union(self, others: Iterable[_T] = ()) -> EventedSet[_T]:
        """Return a set containing the union of sets"""
        return type(self)(self._set.union(others))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField):
        """Pydantic validator."""
        from pydantic.utils import sequence_like

        if not sequence_like(v):
            raise TypeError(
                trans._(
                    'Value is not a valid sequence: {value}',
                    deferred=True,
                    value=v,
                )
            )
        if not field.sub_fields:
            return cls(v)

        type_field = field.sub_fields[0]
        errors = []
        for i, v_ in enumerate(v):
            _valid_value, error = type_field.validate(v_, {}, loc=f'[{i}]')
            if error:
                errors.append(error)
        if errors:
            from pydantic import ValidationError

            raise ValidationError(errors, cls)  # type: ignore
        return cls(v)

    def _json_encode(self):
        """Return an object that can be used by json.dumps."""
        return list(self)

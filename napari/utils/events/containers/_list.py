"""MutableSequence that emits events when altered.

Note For Developers
===================

Be cautious when re-implementing typical list-like methods here (e.g. extend,
pop, clear, etc...).  By not re-implementing those methods, we force ALL "CRUD"
(create, read, update, delete) operations to go through a few key methods
defined by the abc.MutableSequence interface, where we can emit the necessary
events.

Specifically:

- ``insert`` = "create" : add a new item/index to the list
- ``__getitem__`` = "read" : get the value of an existing index
- ``__setitem__`` = "update" : update the value of an existing index
- ``__delitem__`` = "delete" : remove an existing index from the list

All of the additional list-like methods are provided by the MutableSequence
interface, and call one of those 4 methods.  So if you override a method, you
MUST make sure that all the appropriate events are emitted.  (Tests should
cover this in test_evented_list.py)
"""

import logging
from typing import (
    Iterable,
    List,
    MutableSequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from ..event import EmitterGroup
from ..types import SupportsEvents

logger = logging.getLogger(__name__)
T = TypeVar('T')
Index = Union[int, slice]


class EventedList(SupportsEvents, MutableSequence[T]):
    """Mutable Sequence that emits events when altered.

    This class is designed to behave exactly like the builtin ``list``, but
    will emit events before and after all mutations (insertion, removal,
    setting, and moving).

    Parameters
    ----------
    data : Iterable, optional
        Initial data, by default None

    Events
    ------
    inserting (index: int)
        emitted before an item is inserted at ``index``
    inserted (index: int, value: T)
        emitted after ``value`` is inserted at ``index``
    removing (index: int)
        emitted before an item is removed at ``index``
    removed (index: int, value: T)
        emitted after ``value`` is removed at ``index``
    moving (index: int, new_index: int)
        emitted before an item is moved from ``index`` to ``new_index``
    moved (index: int, new_index: int, value: T)
        emitted after ``value`` is moved from ``index`` to ``new_index``
    changed (index: int, old_value: T, new_value: T)
        emitted when ``index`` is set from ``old_value`` to ``new_value``
    changed <OVERLOAD> (index: slice, old_value: List[T], new_value: List[T])
        emitted when ``index`` is set from ``old_value`` to ``new_value``
    reordered (value: self)
        emitted when the list is reordered (eg. moved/reversed).
    """

    def __init__(self, data: Iterable[T] = None):
        self._list: List[T] = []
        self.events: EmitterGroup

        _events = {
            'inserting': None,  # int
            'inserted': None,  # Tuple[int, Any] - (idx, value)
            'removing': None,  # int
            'removed': None,  # Tuple[int, Any] - (idx, value)
            'moving': None,  # Tuple[int, int]
            'moved': None,  # Tuple[Tuple[int, int], Any]
            'changed': None,  # Tuple[int, Any, Any] - (idx, old, new)
            'reordered': None,  # None
        }

        # If the superclass already has an EmitterGroup, add to it
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            self.events = EmitterGroup(source=self, **_events)

        if data is not None:
            self.extend(data)

    # WAIT!! ... Read the module docstring before reimplement these classes
    # def append(self, item): ...
    # def clear(self): ...
    # def pop(self, index=-1): ...
    # def extend(self, value: Iterable[T]): ...
    # def remove(self, value: T): ...

    @overload
    def __getitem__(self, key: int) -> T:
        ...

    @overload
    def __getitem__(self, key: slice) -> 'EventedList[T]':  # noqa: F811
        ...

    def __getitem__(self, key):  # noqa: F811 (redefinition)
        result = self._list[key]
        if isinstance(result, list):
            return self.__class__(result)
        return result

    @overload
    def __setitem__(self, key: int, value: T):
        ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]):  # noqa: F811
        ...

    def __setitem__(self, key, value):  # noqa: F811 (redefinition)
        old = self._list[key]
        self._list[key] = value
        if value != old:
            self.events.changed(index=key, old_value=old, new_value=value)

    def _delitem_indices(
        self, key: Index
    ) -> Iterable[Tuple['EventedList[T]', int]]:
        # returning List[(self, int)] allows subclasses to pass nested members
        if isinstance(key, int):
            return [(self, key if key >= 0 else key + len(self))]
        elif isinstance(key, slice):
            _start = key.start or 0
            _stop = key.stop or len(self)
            _step = key.step or 1
            if _start < 0:
                _start = len(self) + _start
            if _stop < 0:
                _stop = len(self) + _stop
            return [(self, i) for i in range(_start, _stop, _step)]
        raise TypeError("Deletion index must be int, or slice")

    def __delitem__(self, key: Index):
        # delete from the end
        for parent, index in sorted(self._delitem_indices(key), reverse=True):
            parent.events.removing(index=index)
            item = parent._list.pop(index)
            parent.events.removed(index=index, value=item)

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return repr(self._list)

    def __eq__(self, other):
        return self._list == other

    def __hash__(self) -> int:
        # it's important to add this to allow this object to be hashable
        # given that we've also reimplemented __eq__
        return id(self)

    def insert(self, index: int, value: T):
        """Insert ``value`` before index."""
        self.events.inserting(index=index)
        self._list.insert(index, value)
        self.events.inserted(index=index, value=value)

    def move(self, cur_index: int, new_index: int) -> bool:
        """Insert object at ``cur_index`` before ``new_index``.

        Both indices refer to the list prior to any object removal
        """
        if new_index > cur_index:
            new_index -= 1

        self.events.moving(index=cur_index, new_index=new_index)
        item = self._list.pop(cur_index)
        self._list.insert(new_index, item)
        self.events.moved(index=cur_index, new_index=new_index, value=item)
        self.events.reordered(value=self)
        return True

    def reverse(self) -> None:
        """Reverse list *IN PLACE*."""
        # reimplementing this method to emit a change event
        # If this method were removed, .reverse() would still be availalbe,
        # it would just emit a "changed" event for each moved index in the list
        self._list.reverse()
        self.events.reordered(value=self)

    def copy(self) -> 'EventedList[T]':
        """Return a shallow copy of the list."""
        return self.__class__(self._list)

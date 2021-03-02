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
from typing import Callable, Dict, Iterable, Sequence, Tuple, Type, Union

from ..event import EmitterGroup
from ._typed import _L, _T, Index, TypedMutableSequence

logger = logging.getLogger(__name__)


class EventedList(TypedMutableSequence[_T]):
    """Mutable Sequence that emits events when altered.

    This class is designed to behave exactly like the builtin ``list``, but
    will emit events before and after all mutations (insertion, removal,
    setting, and moving).

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the list with.
    basetype : type or sequence of types, optional
        Type of the elements in the list.
    lookup : dict of Type[L] : function(object) -> L
        Mapping between a type, and a function that converts items in the list
        to that type.

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
    changed (index: int, old_value: T, value: T)
        emitted when ``index`` is set from ``old_value`` to ``value``
    changed <OVERLOAD> (index: slice, old_value: List[_T], value: List[_T])
        emitted when ``index`` is set from ``old_value`` to ``value``
    reordered (value: self)
        emitted when the list is reordered (eg. moved/reversed).
    """

    events: EmitterGroup

    def __init__(
        self,
        data: Iterable[_T] = (),
        *,
        basetype: Union[Type[_T], Sequence[Type[_T]]] = (),
        lookup: Dict[Type[_L], Callable[[_T], Union[_T, _L]]] = dict(),
    ):
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

        # For inheritance: If the mro already provides an EmitterGroup, add...
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            # otherwise create a new one
            self.events = EmitterGroup(source=self, **_events)
        super().__init__(data, basetype=basetype, lookup=lookup)

    # WAIT!! ... Read the module docstring before reimplement these methods
    # def append(self, item): ...
    # def clear(self): ...
    # def pop(self, index=-1): ...
    # def extend(self, value: Iterable[_T]): ...
    # def remove(self, value: T): ...

    def __setitem__(self, key, value):
        old = self._list[key]
        if value is old:  # https://github.com/napari/napari/pull/2120
            return
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError('Can only assign an iterable to slice')
            [self._type_check(v) for v in value]  # before we mutate the list
            if key.step is not None:  # extended slices are more restricted
                indices = list(range(*key.indices(len(self))))
                if not len(value) == len(indices):
                    raise ValueError(
                        f"attempt to assign sequence of size {len(value)} to "
                        f"extended slice of size {len(indices)}"
                    )
                for i, v in zip(indices, value):
                    self.__setitem__(i, v)
            else:
                del self[key]
                start = key.start or 0
                for i, v in enumerate(value):
                    self.insert(start + i, v)
        else:
            super().__setitem__(key, value)
            self.events.changed(index=key, old_value=old, value=value)

    def _delitem_indices(
        self, key: Index
    ) -> Iterable[Tuple['EventedList[_T]', int]]:
        # returning List[(self, int)] allows subclasses to pass nested members
        if isinstance(key, int):
            return [(self, key if key >= 0 else key + len(self))]
        elif isinstance(key, slice):
            return [(self, i) for i in range(*key.indices(len(self)))]
        elif type(key) in self._lookup:
            return [(self, self.index(key))]

        valid = {int, slice}.union(set(self._lookup))
        raise TypeError(f"Deletion index must be {valid!r}, got {type(key)}")

    def __delitem__(self, key: Index):
        # delete from the end
        for parent, index in sorted(self._delitem_indices(key), reverse=True):
            parent.events.removing(index=index)
            item = parent._list.pop(index)
            parent.events.removed(index=index, value=item)

    def insert(self, index: int, value: _T):
        """Insert ``value`` before index."""
        self.events.inserting(index=index)
        super().insert(index, value)
        self.events.inserted(index=index, value=value)

    def move(self, src_index: int, dest_index: int = 0) -> bool:
        """Insert object at ``src_index`` before ``dest_index``.

        Both indices refer to the list prior to any object removal
        (pre-move space).
        """
        if dest_index < 0:
            dest_index += len(self) + 1
        if dest_index > src_index:
            dest_index -= 1

        self.events.moving(index=src_index, dest_index=dest_index)
        item = self._list.pop(src_index)
        self._list.insert(dest_index, item)
        self.events.moved(index=src_index, dest_index=dest_index, value=item)
        self.events.reordered(value=self)
        return True

    def move_multiple(
        self, sources: Sequence[Index], dest_index: int = 0
    ) -> int:
        """Move a batch of indices, to a single destination.

        Note, if the dest_index is higher than any of the source indices, then
        the resulting position of the moved objects after the move operation
        is complete will be lower than ``dest_index``.

        Parameters
        ----------
        sources : Sequence[int or slice]
            A sequence of indices
        dest_index : int, optional
            The destination index.  All sources will be inserted before this
            index (in pre-move space), by default 0... which has the effect of
            "bringing to front" everything in ``sources``, or acting as a
            "reorder" method if ``sources`` contains all indices.

        Returns
        -------
        int
            The number of successful move operations completed.

        Raises
        ------
        TypeError
            If the destination index is a slice, or any of the source indices
            are not ``int`` or ``slice``.
        """
        if isinstance(dest_index, slice):
            raise TypeError("Destination index may not be a slice")

        to_move = []
        for idx in sources:
            if isinstance(idx, slice):
                to_move.extend(list(range(*idx.indices(len(self)))))
            elif isinstance(idx, int):
                to_move.append(idx)
            else:
                raise TypeError("Can only move integer or slice indices")

        # remove duplicates while maintaing order, python 3.7+
        to_move = list(dict.fromkeys(to_move))

        if dest_index < 0:
            dest_index += len(self) + 1
        dest_index -= len([i for i in to_move if i < dest_index])

        self.events.moving(index=to_move, new_index=dest_index)
        items = [self[i] for i in to_move]
        for i in sorted(to_move, reverse=True):
            self._list.pop(i)
        for item in items[::-1]:
            self._list.insert(dest_index, item)
        self.events.moved(index=to_move, new_index=dest_index, value=items)
        self.events.reordered(value=self)
        return len(to_move)

    def reverse(self) -> None:
        """Reverse list *IN PLACE*."""
        # reimplementing this method to emit a change event
        # If this method were removed, .reverse() would still be availalbe,
        # it would just emit a "changed" event for each moved index in the list
        self._list.reverse()
        self.events.reordered(value=self)

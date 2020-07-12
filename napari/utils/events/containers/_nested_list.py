from __future__ import annotations

import logging
from collections import defaultdict
from typing import (
    DefaultDict,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from ..event import Event
from ._list import EventedList, Index, SupportsEvents, T

logger = logging.getLogger(__name__)

NestedIndex = Tuple[Index, ...]


def ensure_tuple_index(index: Union[NestedIndex, Index]) -> NestedIndex:
    if isinstance(index, (slice, int)):
        return (index,)  # single integer inserts to self
    elif isinstance(index, tuple):
        return index
    raise TypeError(f"Invalid nested index: {index}. Must be an int or tuple")


def split_nested_index(
    index: Union[NestedIndex, Index]
) -> Tuple[NestedIndex, Index]:
    """Given a nested index, return (nested_parent_index, row)."""
    index = ensure_tuple_index(index)
    if index:
        *par, i = index
        return tuple(par), i
    return (), -1  # empty tuple appends to self


class NestableEventedList(EventedList[T]):
    """Nestable Mutable Sequence that emits recursive events when altered.

    A key property of this class is that when new mutable sequences are added
    to the list, they are themselves converted to a ``NestableEventedList``,
    and all of the ``EventEmitter`` objects in the child are connect to the
    parent object's ``_reemit_nested_event`` method. When
    ``_reemit_nested_event`` receives an event from a child object, it remits
    the event, but changes any ``index`` keys in the event to a ``NestedIndex``
    (a tuple of ``int``) such that indices emitted by any given
    ``NestableEventedList`` are always relative to itself.

    ``NestableEventedList`` instances can be indexed with a ``tuple`` of
    ``int`` (e.g. ``mylist[0, 2, 1]``) to retrieve nested child objects.

    Parameters
    ----------
    data : Iterable, optional
        Initial data, by default None

    Events
    ------
    types used:
        Index = Union[int, Tuple[int, ...]]

    inserting (index: Index)
        emitted before an item is inserted at ``index``
    inserted (index: Index, value: T)
        emitted after ``value`` is inserted at ``index``
    removing (index: Index)
        emitted before an item is removed at ``index``
    removed (index: Index, value: T)
        emitted after ``value`` is removed at ``index``
    moving (index: Index, new_index: Index)
        emitted before an item is moved from ``index`` to ``new_index``
    moved (index: Index, new_index: Index, value: T)
        emitted after ``value`` is moved from ``index`` to ``new_index``
    changed (index: Index, old_value: T, new_value: T)
        emitted when ``index`` is set from ``old_value`` to ``new_value``
    changed <OVERLOAD> (index: slice, old_value: List[T], new_value: List[T])
        emitted when slice ``index`` is set from ``old_value`` to ``new_value``
    reordered (value: self)
        emitted when the list is reordered (eg. moved/reversed).
    """

    # WAIT!! ... Read the ._list module docs before reimplement these classes
    # def append(self, item): ...
    # def clear(self): ...
    # def pop(self, index=-1): ...
    # def extend(self, value: Iterable[T]): ...
    # def remove(self, value: T): ...

    # fmt: off
    @overload
    def __getitem__(self, key: Union[int, NestedIndex]) -> T: ...  # noqa: E704

    @overload
    def __getitem__(self, key: slice) -> NestableEventedList[T]: ...  # noqa

    def __getitem__(self, key):  # noqa: F811
        if isinstance(key, tuple):
            item: NestableEventedList[T] = self
            for idx in key:
                item = item[idx]
            return item
        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: Union[int, NestedIndex], value: T): ...  # noqa

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]): ...  # noqa
    # fmt: on

    def __setitem__(self, key, value):  # noqa: F811
        # FIXME: !!!!
        # this is delicate, we want to preserve the evented list when nesting
        # but there is a high risk here of clobbering attributes of a special
        # child class
        if isinstance(value, list):
            value = self.__class__(value)
        if isinstance(key, tuple):
            parent_i, index = split_nested_index(key)
            self[parent_i].__setitem__(index, value)
            return
        self._connect_child_emitters(value)
        EventedList.__setitem__(self, key, value)

    @overload
    def _delitem_indices(
        self, key: Index
    ) -> Iterable[Tuple[EventedList[T], int]]:
        ...

    @overload
    def _delitem_indices(  # noqa: F811
        self, key: NestedIndex
    ) -> Iterable[Tuple[EventedList[T], Index]]:
        ...

    def _delitem_indices(self, key):  # noqa: F811
        if isinstance(key, tuple):
            parent_i, index = split_nested_index(key)
            return [(cast(NestableEventedList[T], self[parent_i]), index)]
        elif isinstance(key, (int, slice)):
            return super()._delitem_indices(key)
        raise TypeError("Deletion index must be int, slice, or tuple")

    def __delitem__(self, key: Index):
        # delete from the end
        for parent, index in self._delitem_indices(key):
            self._disconnect_child_emitters(parent[index])
        super().__delitem__(key)

    # TODO: implement __eq__

    def insert(self, index: int, value: T):
        # FIXME: !!!!
        # this is delicate, we want to preserve the evented list when nesting
        # but there is a high risk here of clobbering attributes of a special
        # child class
        if isinstance(value, list):
            value = self.__class__(value)
        super().insert(index, value)
        self._connect_child_emitters(value)

    def _reemit_nested_event(self, event: Event):
        emitter = getattr(self.events, event.type, None)
        if not emitter:
            return

        source_index = self.index(event.source)
        for attr in ('index', 'new_index'):
            if hasattr(event, attr):
                cur_index = ensure_tuple_index(event.index)
                setattr(event, attr, (source_index,) + cur_index)
        if not hasattr(event, 'index'):
            setattr(event, 'index', source_index)

        emitter(event)

    def _disconnect_child_emitters(self, child: T):
        # IMPORTANT!! this is currently assuming that all emitter groups
        # are named "events"
        if isinstance(child, SupportsEvents):
            for emitter in child.events.emitters.values():
                emitter.disconnect(self._reemit_nested_event)

    def _connect_child_emitters(self, child: T):
        # IMPORTANT!! this is currently assuming that all emitter groups
        # are named "events"
        if isinstance(child, SupportsEvents):
            for emitter in child.events.emitters.values():
                emitter.connect(self._reemit_nested_event)

    def _non_negative_index(
        self, parent_index: NestedIndex, dest_index: Index
    ) -> Index:
        destination_group = cast(NestableEventedList[T], self[parent_index])
        if isinstance(dest_index, int):
            if dest_index < 0:
                dest_index += len(destination_group) + 1
            else:
                # TODO: Necessary?
                dest_index = min(dest_index, len(destination_group))
        return dest_index

    def move_multiple(
        self, sources: Sequence[NestedIndex], dest_index: NestedIndex,
    ) -> int:
        """Move a batch of nested indices, to a single destination."""
        logger.debug(
            f"move_multiple(sources={sources}, dest_index={dest_index})"
        )
        dest_par, dest_i = split_nested_index(dest_index)
        if isinstance(dest_i, slice):
            raise ValueError("Destination index may not be a slice")
        dest_i = self._non_negative_index(dest_par, dest_i)
        dest_i = cast(int, dest_i)
        logger.debug(f"destination: {dest_par}[{dest_i}]")

        self.events.reordered.block()
        moved = 0
        # more complicated when moving multiple objects.
        # don't assume index adjacency ... so move one at a time
        # need to update indices as we pop, so we keep track of the indices
        # we have previously popped
        popped: DefaultDict[NestedIndex, List[int]] = defaultdict(list)
        # we iterate indices from the end first, so pop() always works

        for i, idx in enumerate(sorted(sources, reverse=True)):
            if idx == ():
                raise IndexError("Group cannot move itself")
            src_par, src_i = split_nested_index(idx)

            if isinstance(src_i, slice):
                raise ValueError("Terminal source index may not be a slice")

            if src_i < 0:
                src_i += len(cast(NestableEventedList[T], self[src_par]))

            # we need to decrement the src_i by 1 for each time we have
            # previously pulled items out from in front of the src_i
            src_i -= sum(map(lambda x: x <= src_i, popped.get(src_par, [])))
            # we need to decrement the dest_i by 1 for each time we have
            # previously pulled items out from in front of the dest_i
            ddec = sum(map(lambda x: x <= dest_i, popped.get(dest_par, [])))

            # FIXME:
            # there is still a bug and a failing test... if we are moving items
            # from a lower level nested group up to a higher level, and inserting
            # into a position is higher than the *parent* of that nested group
            # we have an index error.  ie:

            # i.e. we need to increase the (src_par, ...) by 1 for each time
            # we have previously inserted items in front of the (src_par, ...)

            # if item is being moved within the same parent,
            # we need to increase the src_i by 1 for each time we have
            # previously inserted items in front of the src_i
            if src_par == dest_par:
                src_i += (dest_i <= src_i) * i
                if src_i == dest_i - ddec:
                    # skip noop
                    continue
            moved += self.move(src_par + (src_i,), dest_par + (dest_i - ddec,))
            popped[src_par].append(src_i)
        self.events.reordered.unblock()
        self.events.reordered(value=self)
        return moved

    def move(
        self,
        cur_index: Union[int, NestedIndex],
        new_index: Union[int, NestedIndex],
    ) -> bool:
        logger.debug(f"move(cur_index={cur_index}, new_index={new_index})")
        src_par_i, src_i = split_nested_index(cur_index)
        dest_par_i, dest_i = split_nested_index(new_index)
        dest_i = self._non_negative_index(dest_par_i, dest_i)
        new_index = dest_par_i + (dest_i,)

        if isinstance(src_i, slice):
            raise ValueError("Terminal source index may not be a slice")

        if src_par_i == dest_par_i:
            if isinstance(dest_i, int):
                if dest_i > src_i:
                    dest_i -= 1
                if src_i == dest_i:
                    return False

        self.events.moving(index=cur_index, new_index=new_index)

        silenced = ['removed', 'removing', 'inserted', 'inserting']
        for event_name in silenced:
            getattr(self.events, event_name).block()

        dest_par = self[dest_par_i]
        value = self[src_par_i].pop(src_i)
        dest_par.insert(dest_i, value)

        for event_name in silenced:
            getattr(self.events, event_name).unblock()

        self.events.moved(index=cur_index, new_index=new_index, value=value)
        self.events.reordered(value=self)
        return True

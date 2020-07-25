"""Nestable MutableSequence that emits events when altered.
"""
import logging
from typing import Iterable, NewType, Sequence, Tuple, Union, cast, overload

from ..event import Event
from ..types import SupportsEvents
from ._list import EventedList, Index, T

logger = logging.getLogger(__name__)

NestedIndex = Tuple[Index, ...]
ParentIndex = NewType('ParentIndex', Tuple[int, ...])


def ensure_tuple_index(index: Union[NestedIndex, Index]) -> NestedIndex:
    """Return index as a tuple of ints or slices.

    Parameters
    ----------
    index : Tuple[Union[int, slice], ...] or int or slice
        An index as an int, tuple, or slice

    Returns
    -------
    NestedIndex
        The index, guaranteed to be a tuple.

    Raises
    ------
    TypeError
        If the input ``index`` is not an ``int``, ``slice``, or ``tuple``.
    """
    if isinstance(index, (slice, int)):
        return (index,)  # single integer inserts to self
    elif isinstance(index, tuple):
        return index
    raise TypeError(f"Invalid nested index: {index}. Must be an int or tuple")


def split_nested_index(
    index: Union[NestedIndex, Index]
) -> Tuple[ParentIndex, Index]:
    """ Given a nested index, return (nested_parent_index, row).

    Parameters
    ----------
    index : Union[NestedIndex, Index]
        An index as an int, tuple, or slice

    Returns
    -------
    Tuple[NestedIndex, Index]
        A tuple of ``parent_index``, ``row``

    Raises
    ------
    ValueError
        If any of the items in the returned ParentIndex tuple are not ``int``.

    Examples
    --------
    >>> split_nested_index((1, 2, 3, 4))
    ((1, 2, 3), 4)
    >>> split_nested_index(1)
    ((), 1)
    >>> split_nested_index(())
    ((), -1)
    """
    index = ensure_tuple_index(index)
    if index:
        *first, last = index
        if any(not isinstance(p, int) for p in first):
            raise ValueError('The parent index must be a tuple of int')
        return cast(ParentIndex, tuple(first)), last
    return ParentIndex(()), -1  # empty tuple appends to self


class NestableEventedList(EventedList[T]):
    """Nestable Mutable Sequence that emits recursive events when altered.

    ``NestableEventedList`` instances can be indexed with a ``tuple`` of
    ``int`` (e.g. ``mylist[0, 2, 1]``) to retrieve nested child objects.

    A key property of this class is that when new mutable sequences are added
    to the list, they are themselves converted to a ``NestableEventedList``,
    and all of the ``EventEmitter`` objects in the child are connect to the
    parent object's ``_reemit_nested_event`` method (assuming the child has
    an attribute called ``events`` that is an instance of ``EmitterGroup``).
    When ``_reemit_nested_event`` receives an event from a child object, it
    remits the event, but changes any ``index`` keys in the event to a
    ``NestedIndex`` (a tuple of ``int``) such that indices emitted by any given
    ``NestableEventedList`` are always relative to itself.


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
    changed (index: Index, old_value: T, value: T)
        emitted when ``index`` is set from ``old_value`` to ``value``
    changed <OVERLOAD> (index: slice, old_value: List[T], value: List[T])
        emitted when slice ``index`` is set from ``old_value`` to ``value``
    reordered (value: self)
        emitted when the list is reordered (eg. moved/reversed).
    """

    # WAIT!! ... Read the ._list module docs before reimplement these classes
    # def append(self, item): ...
    # def clear(self): ...
    # def pop(self, index=-1): ...
    # def extend(self, value: Iterable[T]): ...
    # def remove(self, value: T): ...

    @overload
    def __getitem__(self, key: int) -> T:
        ...  # pragma: no cover

    @overload
    def __getitem__(  # noqa: F811
        self, key: ParentIndex
    ) -> 'NestableEventedList[T]':
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: slice) -> 'NestableEventedList[T]':  # noqa
        ...  # pragma: no cover

    @overload
    def __getitem__(  # noqa: F811
        self, key: NestedIndex
    ) -> Union[T, 'NestableEventedList[T]']:
        ...  # pragma: no cover

    def __getitem__(self, key):  # noqa: F811
        if isinstance(key, tuple):
            item: NestableEventedList[T] = self
            for idx in key:
                item = item[idx]
            return item
        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: Union[int, NestedIndex], value: T):
        ...  # pragma: no cover

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]):  # noqa: F811
        ...  # pragma: no cover

    def __setitem__(self, key, value):  # noqa: F811
        # NOTE: if we check isinstance(..., MutableList), then we'll actually
        # clobber object of specialized classes being inserted into the list
        # (for instance, subclasses of NestableEventedList)
        # this check is more conservative, but will miss some "nestable" things
        if isinstance(value, list):
            value = self.__class__(value)
        if isinstance(key, tuple):
            parent_i, index = split_nested_index(key)
            self[parent_i].__setitem__(index, value)
            return
        self._connect_child_emitters(value)
        super().__setitem__(key, value)

    @overload
    def _delitem_indices(
        self, key: Index
    ) -> Iterable[Tuple[EventedList[T], int]]:
        ...  # pragma: no cover

    @overload
    def _delitem_indices(  # noqa: F811
        self, key: NestedIndex
    ) -> Iterable[Tuple[EventedList[T], Index]]:
        ...  # pragma: no cover

    def _delitem_indices(self, key):  # noqa: F811
        if isinstance(key, tuple):
            parent_i, index = split_nested_index(key)
            return [(cast(NestableEventedList[T], self[parent_i]), index)]
        elif isinstance(key, (int, slice)):
            return super()._delitem_indices(key)
        raise TypeError("Deletion index must be int, slice, or tuple")

    def __delitem__(self, key):
        # delete from the end
        for parent, index in self._delitem_indices(key):
            self._disconnect_child_emitters(parent[index])
        super().__delitem__(key)

    # TODO: implement __eq__

    def insert(self, index: int, value: T):
        """Insert object before index."""
        # this is delicate, we want to preserve the evented list when nesting
        # but there is a high risk here of clobbering attributes of a special
        # child class
        if isinstance(value, list):
            value = self.__class__(value)
        super().insert(index, value)
        self._connect_child_emitters(value)

    def _reemit_nested_event(self, event: Event):
        source_index = self.index(event.source)
        for attr in ('index', 'new_index'):
            if hasattr(event, attr):
                cur_index = ensure_tuple_index(event.index)
                setattr(event, attr, (source_index,) + cur_index)
        if not hasattr(event, 'index'):
            setattr(event, 'index', source_index)

        # reemit with this object's EventEmitter of the same type if present
        # otherwise just emit with the EmitterGroup itself
        getattr(self.events, event.type, self.events)(event)

    def _disconnect_child_emitters(self, child: T):
        """Disconnect all events from the child from the reemitter."""
        if isinstance(child, SupportsEvents):
            child.events.disconnect(self._reemit_nested_event)

    def _connect_child_emitters(self, child: T):
        """Connect all events from the child to be reemitted."""
        if isinstance(child, SupportsEvents):
            # make sure the event source has been set on the child
            if child.events.source is None:
                child.events.source = child
            child.events.connect(self._reemit_nested_event)

    def _non_negative_index(
        self, parent_index: ParentIndex, dest_index: Index
    ) -> Index:
        """Make sure dest_index is a positive index inside parent_index."""
        destination_group = cast(NestableEventedList[T], self[parent_index])
        # not handling slice indexes
        if isinstance(dest_index, int):
            if dest_index < 0:
                dest_index += len(destination_group) + 1
        return dest_index

    def move_multiple(
        self, sources: Sequence[NestedIndex], dest_index: NestedIndex,
    ) -> int:
        """Move a batch of nested indices, to a single destination.

        This handles the complications of changing the removal and insertion
        indices while poping and inserting items from arbitrary nested
        locations in the tree.

        Parameters
        ----------
        sources : Sequence[NestedIndex]
            A sequence of indices in nested index form.
        dest_index : NestedIndex
            The destination index.  All sources will be inserted before this
            index.

        Returns
        -------
        int
            The number of successful move operations completed.

        Raises
        ------
        ValueError
            If either the destination index or one of the terminal source
            indices are ``slice``.
        IndexError
            If one of the source indices is this group itself.
        """
        logger.debug(
            f"move_multiple(sources={sources}, dest_index={dest_index})"
        )
        dest_par, dest_i = split_nested_index(dest_index)
        if isinstance(dest_i, slice):
            raise ValueError("Destination index may not be a slice")
        dest_i = self._non_negative_index(dest_par, dest_i)
        dest_i = cast(int, dest_i)
        logger.debug(f"destination: {dest_par}[{dest_i}]")

        moved = 0

        _store = []
        shift_dest: int = 0
        # first make an intermediate list of all the objects we're moving
        for idx in sources:
            if idx == ():
                raise IndexError("Group cannot move itself")
            src_par, src_i = split_nested_index(idx)
            if isinstance(src_i, slice):
                raise ValueError("Terminal source index may not be a slice")
            _store.append(self[idx])
            # we need to decrement the destination index by 1 for each time we
            # pull items in front of dest_i from the same parent as the dest
            if src_par == dest_par and src_i < dest_i:
                shift_dest -= 1

        # TODO: add the appropriate moving/moved events
        with self.events.blocker():
            # delete the stored items from the list
            for idx in sorted(sources, reverse=True):
                del self[idx]
            dest_i += shift_dest
            # insert into the destination
            self[dest_par][dest_i:dest_i] = _store

        self.events.reordered(value=self)
        return moved

    def move(
        self,
        cur_index: Union[int, NestedIndex],
        new_index: Union[int, NestedIndex],
    ) -> bool:
        """Move a single item from ``cur_index`` to ``new_index``.

        Parameters
        ----------
        cur_index : Union[int, NestedIndex]
            The index of the object to move
        new_index : Union[int, NestedIndex]
            The destination.  Object will be inserted before ``new_index.``

        Returns
        -------
        bool
            Whether the operation completed successfully

        Raises
        ------
        ValueError
            If the terminal source is a slice, or if the source is this root
            object
        """
        logger.debug(f"move(cur_index={cur_index}, new_index={new_index})")
        src_par_i, src_i = split_nested_index(cur_index)
        dest_par_i, dest_i = split_nested_index(new_index)
        dest_i = self._non_negative_index(dest_par_i, dest_i)
        new_index = dest_par_i + (dest_i,)

        if isinstance(src_i, slice):
            raise ValueError("Terminal source index may not be a slice")
        if isinstance(dest_i, slice):
            raise ValueError("Destination index may not be a slice")
        if src_i == ():
            raise ValueError("Group cannot move itself")

        if src_par_i == dest_par_i:
            if isinstance(dest_i, int):
                if dest_i > src_i:
                    dest_i -= 1
                if src_i == dest_i:
                    return False

        self.events.moving(index=cur_index, new_index=new_index)
        with self.events.blocker():
            dest_par = self[dest_par_i]  # grab this before popping src_i
            value = self[src_par_i].pop(src_i)
            dest_par.insert(dest_i, value)

        self.events.moved(index=cur_index, new_index=new_index, value=value)
        self.events.reordered(value=self)
        return True

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
from contextlib import contextmanager
from itertools import zip_longest
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Type, Union

from ...translations import trans
from ..event import EmitterGroup, Event
from ..types import SupportsEvents
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

        self._parent = None
        self._validate = True
        super().__init__(data, basetype=basetype, lookup=lookup)

    # WAIT!! ... Read the module docstring before reimplement these methods
    # def append(self, item): ...
    # def clear(self): ...
    # def pop(self, index=-1): ...
    # def extend(self, value: Iterable[_T]): ...
    # def remove(self, value: T): ...

    def __setitem__(self, key, value):
        tmp = self._list.copy()
        tmp[key] = value
        value = self._validate_with_parent(tmp)[key]

        old = self._list[key]  # https://github.com/napari/napari/pull/2120
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError(
                    trans._(
                        'Can only assign an iterable to slice',
                        deferred=True,
                    )
                )
            if all(
                new_el is old_el for new_el, old_el in zip_longest(value, old)
            ):
                return

            [self._type_check(v) for v in value]  # before we mutate the list
            if key.step is not None:  # extended slices are more restricted
                indices = list(range(*key.indices(len(self))))
                if not len(value) == len(indices):
                    raise ValueError(
                        trans._(
                            "attempt to assign sequence of size {size} to extended slice of size {slice_size}",
                            deferred=True,
                            size=len(value),
                            slice_size=len(indices),
                        )
                    )
                for i, v in zip(indices, value):
                    self.__setitem__(i, v)
            else:
                with self.events.blocker_all():
                    old = self[key]
                    del self[key]
                    start = key.start or 0
                    for i, v in enumerate(value):
                        self.insert(start + i, v)
                self.events.changed(index=key, old_value=old, value=value)
        else:
            if value is old:
                return
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
        raise TypeError(
            trans._(
                "Deletion index must be {valid!r}, got {dtype}",
                deferred=True,
                valid=valid,
                dtype=type(key),
            )
        )

    def __delitem__(self, key: Index):
        tmp = self._list.copy()
        del tmp[key]
        self._validate_with_parent(tmp)
        # delete from the end
        for parent, index in sorted(self._delitem_indices(key), reverse=True):
            parent.events.removing(index=index)
            self._disconnect_child_emitters(parent[index])
            item = parent._list.pop(index)
            self._process_delete_item(item)
            parent.events.removed(index=index, value=item)

    def _process_delete_item(self, item):
        """Allow process item in inherited class before event was emitted"""

    def insert(self, index: int, value: _T):
        """Insert ``value`` before index."""
        tmp = self._list.copy()
        tmp.insert(index, value)
        value = self._validate_with_parent(tmp)[index]
        self.events.inserting(index=index)
        super().insert(index, value)
        self.events.inserted(index=index, value=value)
        self._connect_child_emitters(value)

    def _reemit_child_event(self, event: Event):
        """An item in the list emitted an event.  Re-emit with index"""
        if not hasattr(event, 'index'):
            try:
                setattr(event, 'index', self.index(event.source))
            except ValueError:
                pass
        # reemit with this object's EventEmitter of the same type if present
        # otherwise just emit with the EmitterGroup itself
        getattr(self.events, event.type, self.events)(event)

    def _disconnect_child_emitters(self, child: _T):
        """Disconnect all events from the child from the reemitter."""
        if isinstance(child, SupportsEvents):
            child.events.disconnect(self._reemit_child_event)

    def _connect_child_emitters(self, child: _T):
        """Connect all events from the child to be reemitted."""
        if isinstance(child, SupportsEvents):
            # make sure the event source has been set on the child
            if child.events.source is None:
                child.events.source = child
            child.events.connect(self._reemit_child_event)

    def move(self, src_index: int, dest_index: int = 0) -> bool:
        """Insert object at ``src_index`` before ``dest_index``.

        Both indices refer to the list prior to any object removal
        (pre-move space).
        """
        if dest_index < 0:
            dest_index += len(self) + 1
        if dest_index in (src_index, src_index + 1):
            # this is a no-op
            return False

        self.events.moving(index=src_index, new_index=dest_index)
        item = self._list.pop(src_index)
        if dest_index > src_index:
            dest_index -= 1
        self._list.insert(dest_index, item)
        self.events.moved(index=src_index, new_index=dest_index, value=item)
        self.events.reordered(value=self)
        return True

    def move_multiple(
        self, sources: Iterable[Index], dest_index: int = 0
    ) -> int:
        """Move a batch of `sources` indices, to a single destination.

        Note, if `dest_index` is higher than any of the `sources`, then
        the resulting position of the moved objects after the move operation
        is complete will be lower than `dest_index`.

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
        logger.debug(
            f"move_multiple(sources={sources}, dest_index={dest_index})"
        )

        # calling list here makes sure that there are no index errors up front
        move_plan = list(self._move_plan(sources, dest_index))

        # don't assume index adjacency ... so move objects one at a time
        # this *could* be simplified with an intermediate list ... but this way
        # allows any views (such as QtViews) to update themselves more easily.
        # If this needs to be changed in the future for performance reasons,
        # then the associated QtListView will need to changed from using
        # `beginMoveRows` & `endMoveRows` to using `layoutAboutToBeChanged` &
        # `layoutChanged` while *manually* updating model indices with
        # `changePersistentIndexList`.  That becomes much harder to do with
        # nested tree-like models.
        with self.events.reordered.blocker():
            for src, dest in move_plan:
                self.move(src, dest)

        self.events.reordered(value=self)
        return len(move_plan)

    def _move_plan(self, sources: Iterable[Index], dest_index: int):
        """Prepared indices for a multi-move.

        Given a set of ``sources`` from anywhere in the list,
        and a single ``dest_index``, this function computes and yields
        ``(from_index, to_index)`` tuples that can be used sequentially in
        single move operations.  It keeps track of what has moved where and
        updates the source and destination indices to reflect the model at each
        point in the process.

        This is useful for a drag-drop operation with a QtModel/View.

        Parameters
        ----------
        sources : Iterable[tuple[int, ...]]
            An iterable of tuple[int] that should be moved to ``dest_index``.
        dest_index : Tuple[int]
            The destination for sources.
        """
        if isinstance(dest_index, slice):
            raise TypeError(
                trans._(
                    "Destination index may not be a slice",
                    deferred=True,
                )
            )

        to_move: List[int] = []
        for idx in sources:
            if isinstance(idx, slice):
                to_move.extend(list(range(*idx.indices(len(self)))))
            elif isinstance(idx, int):
                to_move.append(idx)
            else:
                raise TypeError(
                    trans._(
                        "Can only move integer or slice indices",
                        deferred=True,
                    )
                )

        to_move = list(dict.fromkeys(to_move))

        if dest_index < 0:
            dest_index += len(self) + 1

        d_inc = 0
        popped: List[int] = []
        for i, src in enumerate(to_move):
            if src != dest_index:
                # we need to decrement the src_i by 1 for each time we have
                # previously pulled items out from in front of the src_i
                src -= sum(x <= src for x in popped)
                # if source is past the insertion point, increment src for each
                # previous insertion
                if src >= dest_index:
                    src += i
                yield src, dest_index + d_inc

            popped.append(src)
            # if the item moved up, icrement the destination index
            if dest_index <= src:
                d_inc += 1

    def reverse(self) -> None:
        """Reverse list *IN PLACE*."""
        # reimplementing this method to emit a change event
        # If this method were removed, .reverse() would still be available,
        # it would just emit a "changed" event for each moved index in the list
        self._list.reverse()
        self.events.reordered(value=self)

    def _update_inplace(self, other):
        # inplace updating only happens from parent after validation, so no need to validate here.
        with self._no_validation():
            self[:] = list(other)

    def _validate_with_parent(self, value):
        if self._parent is not None and self._validate:
            parent = self._parent[0]
            field = self._parent[1]
            pdict = parent.dict()
            new = value.copy()
            pdict[field] = new
            new = parent._pre_validate(pdict)
            # TODO this actually fails if validation causes a field other than `field` to
            # change; that change won't be upstreamed and we break...
            return new[field]
        else:
            return value

    @contextmanager
    def _no_validation(self):
        self._validate = False
        yield
        self._validate = True

    def _uneventful(self):
        ret = list()
        for el in self:
            if isinstance(el, self.__class__):
                ret.append(el._uneventful())
            else:
                ret.append(el)
        return ret

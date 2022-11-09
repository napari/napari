import warnings
from typing import TypeVar

from napari.utils.events.containers._evented_list import EventedList
from napari.utils.events.containers._nested_list import NestableEventedList
from napari.utils.events.containers._selection import Selectable
from napari.utils.translations import trans

_T = TypeVar("_T")


class SelectableEventedList(Selectable[_T], EventedList[_T]):
    """List model that also supports selection.

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

    selection.changed (added: Set[_T], removed: Set[_T])
        Emitted when the set changes, includes item(s) that have been added
        and/or removed from the set.
    selection.active (value: _T)
        emitted when the current item has changed.
    selection._current (value: _T)
        emitted when the current item has changed. (Private event)
    """

    def __init__(self, *args, **kwargs) -> None:
        self._activate_on_insert = True
        super().__init__(*args, **kwargs)
        self.events.removed.connect(
            lambda e: self.selection.discard(e.value)
        )  # FIXME remove lambda
        self.selection._pre_add_hook = self._preselect_hook

    def _preselect_hook(self, value):
        """Called before adding an item to the selection."""
        if value not in self:
            raise ValueError(
                trans._(
                    "Cannot select item that is not in list: {value!r}",
                    deferred=True,
                    value=value,
                )
            )
        return value

    def insert(self, index: int, value: _T):
        super().insert(index, value)
        if self._activate_on_insert:
            # Make layer selected and unselect all others
            self.selection.active = value

    def select_all(self):
        """Select all items in the list."""
        self.selection.update(self)

    def remove_selected(self):
        """Remove selected items from list."""
        idx = 0
        for i in list(self.selection):
            idx = self.index(i)
            self.remove(i)
        if isinstance(idx, int):
            new = max(0, (idx - 1))
            do_add = len(self) > new
        else:
            *root, _idx = idx
            new = tuple(root) + (_idx - 1,) if _idx >= 1 else tuple(root)
            do_add = len(self) > new[0]
        if do_add:
            self.selection.add(self[new])

    def move_selected(self, index: int, insert: int):
        """Reorder list by moving the item at index and inserting it
        at the insert index. If additional items are selected these will
        get inserted at the insert index too. This allows for rearranging
        the list based on dragging and dropping a selection of items, where
        index is the index of the primary item being dragged, and insert is
        the index of the drop location, and the selection indicates if
        multiple items are being dragged. If the moved layer is not selected
        select it.

        This method is deprecated. Please use layers.move_multiple
        with layers.selection instead.

        Parameters
        ----------
        index : int
            Index of primary item to be moved
        insert : int
            Index that item(s) will be inserted at
        """
        # this is just here for now to support the old layerlist API
        warnings.warn(
            trans._(
                'move_selected is deprecated since 0.4.16. Please use layers.move_multiple with layers.selection instead.',
                deferred=True,
            ),
            FutureWarning,
            stacklevel=2,
        )
        if self[index] not in self.selection:
            self.selection.select_only(self[index])
            moving = [index]
        else:
            moving = [i for i, x in enumerate(self) if x in self.selection]
        offset = insert >= index
        self.move_multiple(moving, insert + offset)

    def select_next(self, step=1, shift=False):
        """Selects next item from list."""
        if self.selection:
            idx = self.index(self.selection._current) + step
            if len(self) > idx >= 0:
                next_layer = self[idx]
                if shift:
                    self.selection.add(next_layer)
                    self.selection._current = next_layer
                else:
                    self.selection.active = next_layer
        elif len(self) > 0:
            self.selection.active = self[-1 if step > 0 else 0]

    def select_previous(self, shift=False):
        """Selects previous item from list."""
        self.select_next(-1, shift=shift)


class SelectableNestableEventedList(
    SelectableEventedList[_T], NestableEventedList[_T]
):
    pass

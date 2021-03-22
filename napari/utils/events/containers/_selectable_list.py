from typing import TypeVar

from ._evented_list import EventedList
from ._selection import Selectable

_T = TypeVar("_T")


class SelectableEventedList(Selectable[_T], EventedList[_T]):
    """List model that also supports selection."""

    def __init__(self, *args, **kwargs) -> None:
        self.activate_on_insert = True
        super().__init__(*args, **kwargs)
        self.events.removed.connect(lambda e: self.selection.discard(e.value))

    # TODO: add strict check to make sure that things added to
    # selection/current are in the list?

    def insert(self, index: int, value: _T):
        super().insert(index, value)
        if self.activate_on_insert:
            # Make layer selected and unselect all others
            self.activate(value)

    def activate(self, value: _T):
        """Set `value` as the current and only selected item."""
        self.selection.select_only(value)
        self.selection.current = value

    def select_all(self):
        """Select all items in the list."""
        self.selection.update(self)

    def remove_selected(self):
        """Remove selected items from list."""
        for i in list(self.selection):
            self.remove(i)

    def select_next(self, shift=False):
        """Selects next item from list."""
        selected_idx = [i for i, x in enumerate(self) if x in self.selection]
        # if anything is selected
        if selected_idx:
            if selected_idx[-1] == len(self) - 1:
                if shift is False:
                    next = self[selected_idx[-1]]
                    self.selection.intersection_update({next})
            elif selected_idx[-1] < len(self) - 1:
                next = self[selected_idx[-1] + 1]
                if shift is False:
                    self.selection.intersection_update({next})
                self.selection.add(next)
        elif len(self) > 0:
            self.selection.add(self[-1])

    def select_previous(self, shift=False):
        """Selects previous item from list."""
        selected_idx = [i for i, x in enumerate(self) if x in self.selection]

        if selected_idx:
            if selected_idx[0] == 0:
                if shift is False:
                    self.selection.intersection_update({self[0]})
            elif selected_idx[0] > 0:
                new = self[selected_idx[0] - 1]
                if shift is False:
                    self.selection.intersection_update({new})
                self.selection.add(new)
        elif len(self) > 0:
            self.selection.add(self[0])

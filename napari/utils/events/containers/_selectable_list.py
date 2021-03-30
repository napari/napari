from typing import TypeVar

from ._evented_list import EventedList
from ._selection import Selectable

_T = TypeVar("_T")


class SelectableEventedList(Selectable[_T], EventedList[_T]):
    """List model that also supports selection."""

    def __init__(self, *args, **kwargs) -> None:
        self._activate_on_insert = True
        super().__init__(*args, **kwargs)
        self.events.removed.connect(lambda e: self.selection.discard(e.value))
        self.selection._pre_add_hook = self._preselect_hook

    def _preselect_hook(self, value):
        """Called before adding an item to the selection."""
        if value not in self:
            raise ValueError(
                f"Cannot select item that is not in list: {value!r}"
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
        for i in list(self.selection):
            self.remove(i)

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

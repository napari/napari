from typing import TypeVar

from ._evented_list import EventedList
from ._selection import Selectable

_T = TypeVar("_T")


class SelectableEventedList(Selectable[_T], EventedList[_T]):
    """List model that also supports selection."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events.removed.connect(lambda e: self.selection.discard(e.value))

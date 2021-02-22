from typing import Iterable, Optional, TypeVar

from ...utils.events import EventedSet

IndexType = TypeVar("IndexType")


class Selection(EventedSet[IndexType]):
    def __init__(self, data: Iterable[IndexType] = ()):
        super().__init__(data=data)
        self.events.add(current=None)
        self._current: Optional[IndexType] = None

    @property
    def current(self) -> Optional[IndexType]:
        return self._current

    @current.setter
    def current(self, index: Optional[IndexType]):
        previous, self._current = self._current, index
        self.events.current(value=index, previous=previous)

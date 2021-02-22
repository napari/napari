from typing import Iterable, Optional, TypeVar

from ...utils.events import EventedSet

_T = TypeVar("_T")


class Selection(EventedSet[_T]):
    def __init__(self, data: Iterable[_T] = ()):
        super().__init__(data=data)
        self.events.add(current=None)
        self._current: Optional[_T] = None

    @property
    def current(self) -> Optional[_T]:
        return self._current

    @current.setter
    def current(self, index: Optional[_T]):
        previous, self._current = self._current, index
        self.events.current(value=index, previous=previous)

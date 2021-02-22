from typing import Iterable, Optional, Tuple, TypeVar, Union

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

    def _validate_item(self, value):
        """May be used by subclasses to provide validation of new values."""
        return value

    def add(self, value: IndexType) -> None:
        return super().add(self._validate_item(value))

    def update(self, others: Iterable[IndexType] = ()) -> None:
        return super().update(others={self._validate_item(o) for o in others})


class ListSelection(Selection[int]):
    def _validate_item(self, value):
        """Make sure that we only add ints."""
        if not isinstance(value, int):
            raise ValueError("Selection index must be int")
        return value


class NestedListSelection(Selection[Union[int, Tuple[int, ...]]]):
    def _validate_item(self, value):
        """Make sure that we only add tuples of ints."""
        if isinstance(value, int):
            return (value,)
        if not isinstance(value, tuple) or any(
            not isinstance(x, int) for x in value
        ):
            raise ValueError("Selection index must be int or tuple of ints")
        return value

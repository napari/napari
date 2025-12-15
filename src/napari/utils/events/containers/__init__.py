from collections.abc import Iterable
from typing import Any, Generic, TypeVar

from psygnal.containers import (
    EventedDict as EventedDict_,
    EventedList as EventedList_,
    EventedSet,
    Selection,
)

from napari.utils.events.containers._typed import (
    TypedLookupSequenceMixin,
    TypedMappingMixin,
)

_T = TypeVar('_T')
_K = TypeVar('_K')


class EventedList(TypedLookupSequenceMixin[_T], EventedList_[_T]):
    pass


class EventedDict(TypedMappingMixin[_K, _T], EventedDict_[_K, _T]):
    pass


class Selectable(Generic[_T]):
    """Mixin that adds a selection model to an object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._selection: Selection[_T] = Selection()
        super().__init__(*args, **kwargs)

    @property
    def selection(self) -> Selection[_T]:
        """Get current selection."""
        return self._selection

    @selection.setter
    def selection(self, new_selection: Iterable[_T]) -> None:
        """Set selection, without deleting selection model object."""
        self._selection.intersection_update(new_selection)
        self._selection.update(new_selection)


class SelectableEventedList(Selectable[_T], EventedList[_T]):
    pass


__all__ = [
    'EventedDict',
    'EventedList',
    'EventedSet',
    'Selectable',
    'SelectableEventedList',
    'Selection',
]

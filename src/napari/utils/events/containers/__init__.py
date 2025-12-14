from psygnal.containers import (
    EventedDict as EventedDict_,
    EventedList as EventedList_,
)

from napari.utils.events.containers._nested_list import NestableEventedList
from napari.utils.events.containers._selectable_list import (
    SelectableEventedList,
    SelectableNestableEventedList,
)
from napari.utils.events.containers._selection import Selectable, Selection
from napari.utils.events.containers._set import EventedSet
from napari.utils.events.containers._typed import (
    TypedLookupSequenceMixin,
    TypedMappingMixin,
)


class EventedList(TypedLookupSequenceMixin, EventedList_):
    pass


class EventedDict(TypedMappingMixin, EventedDict_):
    pass


__all__ = [
    'EventedDict',
    'EventedList',
    'EventedSet',
    'NestableEventedList',
    'Selectable',
    'SelectableEventedList',
    'SelectableNestableEventedList',
    'Selection',
]

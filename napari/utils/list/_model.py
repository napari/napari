from ..event import EmitterGroup, Event
from ..event_handler import EventHandler

from ._multi import MultiIndexList
from ._typed import TypedList


class ListModel(MultiIndexList, TypedList):
    """List with events, tuple-indexing, typing, and filtering.

    Parameters
    ----------
    basetype : type
        Type of the elements in the list.
    iterable : iterable, optional
        Elements to initialize the list with.
    lookup : dict of type : function(object, ``basetype``) -> bool
        Functions that determine if an object is a reference to an
        element of the list.

    Attributes
    ----------
    events : vispy.util.event.EmitterGroup
        Group of events for adding, removing, and reordering elements
        within the list.
    """

    def __init__(self, basetype, iterable=(), lookup=None):
        super().__init__(basetype, iterable, lookup)
        self.events = EmitterGroup(
            source=self,
            auto_connect=False,
            added=Event,
            removed=Event,
            reordered=Event,
            changed=Event,
        )
        self.event_handler = EventHandler(component=self)
        self.events.connect(self.event_handler.on_change)

    def __setitem__(self, query, values):
        new_indices = tuple(self.__prsitem__(query))
        old_indices = tuple(self.index(v) for v in tuple(values))

        if sorted(new_indices) != sorted(old_indices):
            raise TypeError(
                'must be a reordering of indices; '
                'setting of list items not allowed'
            )

        self.events.reordered((old_indices, new_indices))
        self.events.changed(None)

    def insert(self, index, obj):
        self.events.added((obj, index))
        self.events.changed(None)

    def append(self, obj):
        self.events.added((obj, len(self)))
        self.events.changed(None)

    def pop(self, key):
        obj = self[key]
        self.events.removed((obj, key))
        self.events.changed(None)
        return obj

    def clear(self):
        while len(self) > 0:
            obj = self[0]
            self.events.removed((obj, 0))
        self.events.changed(None)

    def reverse(self):
        old_indices = tuple(range(len(self)))
        new_indices_list = list(range(len(self)))
        new_indices_list.reverse()
        new_indices = tuple(new_indices_list)
        self.events.reordered((old_indices, new_indices))
        self.events.changed(None)

    def _on_reordered_change(self, indices):
        old_indices, new_indices = indices
        values = tuple(self[i] for i in old_indices)
        super().__setitem__(new_indices, values)

    def _on_added_change(self, value):
        obj, index = value
        if index == len(self):
            TypedList.append(self, obj)
        else:
            super().insert(index, obj)

    def _on_removed_change(self, value):
        obj, key = value
        super().pop(key)

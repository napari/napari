from ...utils.event import EmitterGroup

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
            auto_connect=True,
            added=None,
            removed=None,
            reordered=None,
            changed=None,
        )
        self.events.added.connect(self.events.changed)
        self.events.removed.connect(self.events.changed)
        self.events.reordered.connect(self.events.changed)

    def __setitem__(self, query, values):
        indices = tuple(self.__prsitem__(query))
        new_indices = tuple(values)

        if sorted(indices) != sorted(self.index(v) for v in new_indices):
            raise TypeError(
                'must be a reordering of indices; '
                'setting of list items not allowed'
            )

        super().__setitem__(indices, new_indices)
        self.events.reordered()

    def insert(self, index, obj):
        super().insert(index, obj)
        self.events.added(item=obj, index=self.__locitem__(index))

    def append(self, obj):
        super(TypedList, self).append(obj)
        self.events.added(item=obj, index=len(self) - 1)

    def pop(self, key):
        obj = super().pop(key)
        self.events.removed(item=obj, index=key)
        return obj

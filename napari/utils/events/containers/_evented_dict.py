"""MutableMapping that emits events when altered."""
from typing import Mapping, Sequence, Type, Union

from ..event import EmitterGroup, Event
from ..types import SupportsEvents
from ._dict import _K, _T, TypedMutableMapping


class EventedDict(TypedMutableMapping[_K, _T]):
    """Mutable dictionary that emits events when altered.

    This class is designed to behave exactly like builting ``dict``, but
    will emit events before and after all mutations (addition, removal, and
    changing).

    Parameters
    ----------
    data : Mapping, optional
        Dictionary to initialize the class with.
    basetype : type of sequence of types, optional
        Type of the element in the dictionary.

    Events
    ------
    changed (key: K, old_value: T, value: T)
        emitted when ``key`` is set from ``old_value`` to ``value``
    adding (key: K)
        emitted before an item is added to the dictionary with ``key``
    added (key: K, value: T)
        emitted after ``value`` was added to the dictionary with ``key``
    removing (key: K)
        emitted before ``key`` is removed from the dictionary
    removed (key: K, value: T)
        emitted after ``key`` was removed from the dictionary
    updated (key, K, value: T)
        emitted after ``value`` of ``key`` was changed. Only implemented by
        subclasses to give them an option to trigger some update after ``value``
        was changed and this class did not register it. This can be useful if
        the ``basetype`` is not an evented object.
    """

    events: EmitterGroup

    def __init__(
        self,
        data: Mapping[_K, _T] = None,
        basetype: Union[Type[_T], Sequence[Type[_T]]] = (),
    ):
        _events = {
            "changing": None,
            "changed": None,
            "adding": None,
            "added": None,
            "removing": None,
            "removed": None,
            "updated": None,
        }
        # For inheritance: If the mro already provides an EmitterGroup, add...
        if hasattr(self, "events") and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            # otherwise create a new one
            self.events = EmitterGroup(source=self, **_events)
        super().__init__(data, basetype)

    def __setitem__(self, key: _K, value: _T):
        old = self._dict.get(key, None)
        if value is old or value == old:
            return
        if old is None:
            self.events.adding(key=key)
            super().__setitem__(key, value)
            self.events.added(key=key, value=value)
            self._connect_child_emitters(value)
        else:
            super().__setitem__(key, value)
            self.events.changed(key=key, old_value=old, value=value)

    def __delitem__(self, key: _K):
        self.events.removing(key=key)
        self._disconnect_child_emitters(self[key])
        item = self._dict.pop(key)
        self.events.removed(key=key, value=item)

    def _reemit_child_event(self, event: Event):
        """An item in the dict emitted an event.  Re-emit with key"""
        if not hasattr(event, "key"):
            setattr(event, "key", self.key(event.source))
        # re-emit with this object's EventEmitter of the same type if present
        # otherwise just emit with the EmitterGroup itself
        getattr(self.events, event.type, self.events)(event)

    def _disconnect_child_emitters(self, child: _T):
        """Disconnect all events from the child from the re-emitter."""
        if isinstance(child, SupportsEvents):
            child.events.disconnect(self._reemit_child_event)

    def _connect_child_emitters(self, child: _T):
        """Connect all events from the child to be re-emitted."""
        if isinstance(child, SupportsEvents):
            # make sure the event source has been set on the child
            if child.events.source is None:
                child.events.source = child
            child.events.connect(self._reemit_child_event)

    def key(self, value: _T):
        """Return first instance of value."""
        for k, v in self._dict.items():
            if v is value or v == value:
                return k

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: Mapping):
        """Pydantic validator."""
        return cls(value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self._dict)})"

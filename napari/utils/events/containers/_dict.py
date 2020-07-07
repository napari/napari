from collections import abc
from ..event import EmitterGroup, Event
from typing import Any, Iterator


_NULL = object()


class EventedDict(abc.MutableMapping):
    def __init__(self, data: dict = None):
        self._dict = dict(data) if data else {}
        self._events = EmitterGroup(
            source=self,
            changed=None,  # List  - Any change at all
            set=None,  # Tuple[int, Any]  - single item has changed
            removed=None,  # Tuple[int, Any]  - item has been removed
        )
        self._events.connect(self._on_event)

    def _on_event(self, event: Event = None):
        if getattr(event, 'type', None) != 'changed':
            self._events.changed(self._dict)

    def __getitem__(self, key: Any):
        return self._dict.__getitem__(key)

    def __setitem__(self, key: Any, value: Any):
        prev = self._dict.get(key, _NULL)
        self._dict.__setitem__(key, value)
        if value != prev:
            self._events.set((key, value))

    def __delitem__(self, key: Any):
        item = self._dict.pop(key)
        self._events.removed((key, item))

    def __iter__(self) -> Iterator:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

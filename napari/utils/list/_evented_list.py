from collections import abc
from typing import Sequence, Any
from napari.utils.event import EmitterGroup


class EventedList(abc.MutableSequence):
    def __init__(self, data: Sequence = None):
        self._list = data or []

        _events = {
            'added': None,  # List[Tuple[int, Any]]  - item(s) have been added
            'removed': None,  # Tuple[int, Any]  - item has been removed
            'set': None,  # Tuple[int, Any]  - single item has changed
            'changed': None,  # List  - Any change at all
        }
        # For multiple inheritance
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            self.events = EmitterGroup(source=self, **_events)
        # if we directly connect with self.events.connect... too many
        # events are added to the event.sources list
        for emitter in self.events.emitters.values():
            emitter.connect(self._emit_changed)

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def __setitem__(self, key, value):
        self._list.__setitem__(key, value)
        self.events.set((key, value))

    def __delitem__(self, key):
        item = self._list.pop(key)
        if key < 0:
            # always emit a positive index
            key += len(self._list) + 1
        self.events.removed((key, item))

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return repr(self._list)

    def insert(self, index: int, value: Any):
        self._list.insert(index, value)
        self.events.added([(index, value)])
        self._connect_child_emitters(value)

    def move(self, old_index, new_index):
        if new_index > old_index:
            new_index -= 1
        self.insert(new_index, self.pop(old_index))

    def reverse(self) -> None:
        # reimplementing to emit a change event
        self._list.reverse()
        self._emit_changed()

    def _emit_changed(self, event=None):
        if getattr(event, 'type', None) not in ('changed', 'bubbled'):
            self.events.changed(self._list)

    def _bubble_event(self, event):
        if event.source != self and event.type != 'bubble':
            with self.events.changed.blocker():
                emitter = getattr(self.events, event.type, None)
                if emitter:
                    emitter(event)

    def _connect_child_emitters(self, child):
        if hasattr(child, 'events') and isinstance(child.events, EmitterGroup):
            for emitter in child.events.emitters.values():
                emitter.connect(self._bubble_event)

    # by disabling these to overrides, all additions go through insert

    # def extend(self, values: Iterable):
    #     """extend sequence by appending elements from the iterable."""
    #     # reimplementing to emit a single event
    #     n = len(self._list)
    #     self._list.extend(values)
    #     idx = []
    #     for i, v in enumerate(values):
    #         idx.append((n + i, v))
    #         self._connect_child_emitters(v)
    #     self.events.added(idx)

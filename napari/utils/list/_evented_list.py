from typing import List, Iterable, MutableSequence, TypeVar, Union, overload
from napari.utils.event import EmitterGroup, Event

T = TypeVar('T')


class EventedList(MutableSequence[T]):
    """Mutable Sequence that emits events when altered

    To avoid emitting events, directly access EventedList._list
    """

    def __init__(self, data: Iterable[T] = None):
        self._list: List[T] = []
        self.events: EmitterGroup

        _events = {
            'added': None,  # List[Tuple[int, Any]] - [(idx, value),]
            'removed': None,  # List[Tuple[int, Any]] - [(idx, value),]
            'changed': None,  # List[Tuple[int, Any, Any]] - [(idx, old, new),]
            'reordered': None,  # None
        }
        # For multiple inheritance
        if hasattr(self, 'events') and isinstance(self.events, EmitterGroup):
            self.events.add(**_events)
        else:
            self.events = EmitterGroup(source=self, **_events)

        if data is not None:
            self.extend(data)

    # fmt: off
    @overload
    def __getitem__(self, key: int) -> T: ...  # noqa: E704

    @overload
    def __getitem__(self, key: slice) -> List[T]: ...  # noqa

    def __getitem__(self, key):  # noqa: F811
        return self._list[key]

    @overload
    def __setitem__(self, key: int, value: T): ...  # noqa: E704

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]): ...  # noqa
    # fmt: on

    def __setitem__(self, key, value):  # noqa: F811
        old = self._list[key]
        self._list[key] = value
        self.events.changed([(key, old, value)])

    def __delitem__(self, key: Union[int, slice]):
        indices: Iterable[int]

        if isinstance(key, int):
            indices = [key if key >= 0 else key + len(self)]
        elif isinstance(key, slice):
            _start = key.start or 0
            _stop = key.stop or len(self)
            _step = key.step or 1
            if _start < 0:
                _start = len(self) + _start
            if _stop < 0:
                _stop = len(self) + _stop
            indices = sorted(range(_start, _stop, _step), reverse=True)

        removed = [(i, self._list.pop(i)) for i in indices]
        if isinstance(key, slice):
            removed.reverse()
        self.events.removed(removed)

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return repr(self._list)

    def insert(self, index: int, value: T):
        self._list.insert(index, value)
        self.events.added([(index, value)])

    def extend(self, values: Iterable[T]):
        """extend sequence by appending elements from the iterable."""
        # reimplementing to emit a single event
        _len = len(self._list)
        self._list.extend(values)
        self.events.added([(_len + i, v) for i, v in enumerate(values)])

    def move(self, old_index: int, new_index: int):
        """No event emitted."""
        if new_index > old_index:
            new_index -= 1
        self._list.insert(new_index, self._list.pop(old_index))
        self.events.reordered()

    def reverse(self) -> None:
        # reimplementing to emit a change event
        self._list.reverse()
        self.events.reordered()


class NestableEventedList(EventedList):
    def insert(self, index: int, value: T):
        super().insert(index, value)
        self._connect_child_emitters(value)

    def extend(self, values: Iterable[T]):
        super().extend(values)
        for v in values:
            self._connect_child_emitters(v)

    def _bubble_event(self, event: Event):
        if event.source != self:
            emitter = getattr(self.events, event.type, None)
            if emitter:
                emitter(event)

    def _connect_child_emitters(self, child: T):
        if isinstance(child, EventedList):
            for emitter in child.events.emitters.values():
                emitter.connect(self._bubble_event)

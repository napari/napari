from typing import Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class EventedMutable(Protocol):
    def events(self) -> EmitterGroup:
        ...

    def _update_inplace(self, other) -> None:
        ...

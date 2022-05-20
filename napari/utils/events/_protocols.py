from typing import Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class Evented(Protocol):
    def events(self) -> EmitterGroup:
        ...


class EventedMutable(Evented):
    _parent = tuple['EventedModel', str]

    def _update_inplace(self) -> None:
        ...

    def _uneventful(self) -> None:
        ...

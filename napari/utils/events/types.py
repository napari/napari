from typing import Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class SupportsEvents(Protocol):
    events: EmitterGroup

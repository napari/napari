from typing_extensions import Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class SupportsEvents(Protocol):
    events: EmitterGroup

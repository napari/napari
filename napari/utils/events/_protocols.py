from typing import Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class Evented(Protocol):
    events: EmitterGroup

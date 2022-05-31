from typing import Any, Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class EventedMutable(Protocol):
    events: EmitterGroup

    def _update_inplace(self, other: Any) -> None:
        ...

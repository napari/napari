from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from .event import EmitterGroup

if TYPE_CHECKING:
    from .evented_model import EventedModel


@runtime_checkable
class Evented(Protocol):
    events: EmitterGroup


@runtime_checkable
class EventedMutable(Evented, Protocol):
    _parent: Optional[tuple['EventedModel', str]]

    def _update_inplace(self) -> None:
        ...

    def _uneventful(self) -> Any:
        ...

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from .event import EmitterGroup

if TYPE_CHECKING:
    from .evented_model import EventedModel


@runtime_checkable
class EventedMutable(Protocol):
    events: EmitterGroup
    _parent: Optional[Tuple['EventedModel', str]]

    def _update_inplace(self, other: Any) -> None:
        ...

    def _uneventful(self) -> Any:
        ...

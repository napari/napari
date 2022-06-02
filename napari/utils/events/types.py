from typing import Any

from typing_extensions import Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class SupportsEvents(Protocol):
    # note that if this gets an object with `events`, it will pass
    # regardless of what it is because type hints are not checked!
    # this happened with some dask objects, though I cannot recall which...
    events: EmitterGroup


class EventedMutable(SupportsEvents, Protocol):
    def _update_inplace(self, other: Any) -> None:
        ...

    def _uneventful(self) -> Any:
        ...

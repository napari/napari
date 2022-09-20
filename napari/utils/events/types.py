from typing import Any, Protocol, runtime_checkable

from .event import EmitterGroup


@runtime_checkable
class SupportsEvents(Protocol):
    # note that if this gets *any* object with an `events` attribute, it will pass
    # always because type hints are not checked!
    # this happened with some dask objects, so this is a weak spot in some cases
    events: EmitterGroup


class EventedMutable(SupportsEvents, Protocol):
    def _update_inplace(self, other: Any) -> None:
        """
        Update inplace the contents of the EventedMutable to match `other`.
        """

    def _uneventful(self) -> Any:
        """
        Return a non-evented version of self. For example:
        - EventedList -> List
        - EventedDict and EventedModel -> Dict
        - ...
        """
        ...

from typing import Protocol, runtime_checkable

from napari.utils.events.event import EmitterGroup


@runtime_checkable
class SupportsEvents(Protocol):
    events: EmitterGroup

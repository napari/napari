from copy import copy

from napari.utils.events import EmitterGroup, Event
from napari.utils.events.event_utils import transfer_connections


def _event_callback(event: Event):
    pass


class EventedObject:
    def __init__(self):
        self.events = EmitterGroup(self, a=Event, b=Event, c=Event)
        # Event a is auto-connected to on_a, but setup the connection
        # from b to _on_b_changed manually.
        self.events.b.connect(self._on_b_changed)
        # Event c is an event that is triggered by a and b.
        self.events.a.connect(self.events.c)
        self.events.b.connect(self.events.c)

    def on_a(self, event: Event):
        pass

    def _on_b_changed(self, event: Event):
        pass


def test_transfer_connections_with_no_external_connections_then_no_change():
    old_object = EventedObject()
    new_object = EventedObject()
    a_cbs_before = copy(new_object.events.a.callbacks)
    b_cbs_before = copy(new_object.events.b.callbacks)
    c_cbs_before = copy(new_object.events.c.callbacks)

    transfer_connections(old_object.events, new_object.events)

    assert new_object.events.a.callbacks == a_cbs_before
    assert new_object.events.b.callbacks == b_cbs_before
    assert new_object.events.c.callbacks == c_cbs_before


def test_transfer_connections_with_external_connections_then_added():
    old_object = EventedObject()
    old_object.events.a.connect(_event_callback)
    old_object.events.c.connect(_event_callback)
    new_object = EventedObject()
    a_cbs_before = copy(new_object.events.a.callbacks)
    b_cbs_before = copy(new_object.events.b.callbacks)
    c_cbs_before = copy(new_object.events.c.callbacks)

    transfer_connections(old_object.events, new_object.events)

    assert new_object.events.a.callbacks == (_event_callback,) + a_cbs_before
    assert new_object.events.b.callbacks == b_cbs_before
    assert new_object.events.c.callbacks == (_event_callback,) + c_cbs_before


def test_transfer_connections_from_emitter_group_then_added():
    old_object = EventedObject()
    old_object.events.connect(_event_callback)
    new_object = EventedObject()
    cbs_before = new_object.events.callbacks

    transfer_connections(old_object.events, new_object.events)

    assert new_object.events.callbacks == (_event_callback,) + cbs_before

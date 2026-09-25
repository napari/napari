import gc
from dataclasses import dataclass
from unittest.mock import Mock

from pydantic import Field

from napari.utils.events import EventedList, EventedModel, disconnect_events
from napari.utils.events.event import Event, EventEmitter
from napari.utils.events.event_utils import (
    _disconnect_all_events,
    connect_no_arg,
    connect_setattr,
    connect_setattr_value,
)


def test_connect_no_arg():
    mock = Mock(['meth'])
    emitter = EventEmitter()
    connect_no_arg(emitter, mock, 'meth')
    emitter(type_name='a', value=1)
    mock.meth.assert_called_once_with()
    assert len(emitter.callbacks) == 1
    del mock
    gc.collect()
    assert len(emitter.callbacks) == 1
    emitter(type_name='a', value=1)
    assert len(emitter.callbacks) == 0


def test_connect_setattr_value():
    mock = Mock()
    emitter = EventEmitter()
    connect_setattr_value(emitter, mock, 'meth')
    emitter(type_name='a', value=1)
    assert mock.meth == 1
    assert len(emitter.callbacks) == 1
    del mock
    gc.collect()
    assert len(emitter.callbacks) == 1
    emitter(type_name='a', value=1)
    assert len(emitter.callbacks) == 0


def test_connect_setattr():
    mock = Mock()
    emitter = EventEmitter()
    connect_setattr(emitter, mock, 'meth')
    emitter(type_name='a', value=1)
    assert isinstance(mock.meth, Event)
    assert mock.meth.value == 1
    assert len(emitter.callbacks) == 1
    del mock
    gc.collect()
    assert len(emitter.callbacks) == 1
    emitter(type_name='a', value=1)
    assert len(emitter.callbacks) == 0


class SubModel(EventedModel):
    y: int = 0


class MyModel(EventedModel):
    x: int = 0
    sub: SubModel = Field(default_factory=SubModel)
    ls: EventedList = Field(default_factory=lambda: EventedList([SubModel()]))

    @property
    def p(self):
        return self.x + 1


@dataclass
class MyMock:
    # mock does not work here cause it creates methods on the fly and
    # they won't be detected by our machinery
    all_calls = 0
    x_calls = 0
    p_calls = 0
    sub_calls = 0
    sub_all_calls = 0
    y_calls = 0
    ls_all_calls = 0
    ls_sub_calls = 0

    def all(self):
        self.all_calls += 1

    def x(self):
        self.x_calls += 1

    def p(self):
        self.p_calls += 1

    def sub(self):
        self.sub_calls += 1

    def sub_all(self):
        self.sub_all_calls += 1

    def y(self):
        self.y_calls += 1

    def ls_all(self):
        self.ls_all_calls += 1

    def ls_sub(self):
        self.ls_sub_calls += 1


def test_disconnect_events_works():
    mock = MyMock()
    model = MyModel()
    model.events.connect(mock.all)
    model.events.x.connect(mock.x)
    model.events.p.connect(mock.p)
    model.events.s.connect(mock.sub)
    model.sub.events.connect(mock.sub_all)
    model.sub.events.y.connect(mock.y)
    model.ls.events.connect(mock.ls_all)
    model.ls[0].events.connect(mock.ls_sub)

    # check top-level events

    model.x = 1
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1

    # only events from one emittergroup are disconnected
    disconnect_events(model.events, mock)

    model.x = 2
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1

    # now do the same but for a child event
    model.sub.y = 1
    assert mock.sub_all_calls == 1
    assert mock.y_calls == 1

    disconnect_events(model.sub.events, mock)

    model.sub.y = 2
    assert mock.sub_all_calls == 1
    assert mock.y_calls == 1

    # now with event from a container
    model.ls[0].y = 1
    assert mock.ls_all_calls == 1
    assert mock.ls_sub_calls == 1

    disconnect_events(model.ls.events, mock)
    disconnect_events(model.ls[0].events, mock)

    model.ls[0].y = 2
    assert mock.ls_all_calls == 1
    assert mock.ls_sub_calls == 1

    # check everything is the same, no spilling across events
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.sub_calls == 0
    assert mock.sub_all_calls == 1
    assert mock.y_calls == 1
    assert mock.ls_all_calls == 1
    assert mock.ls_sub_calls == 1


def test_disconnect_all_events_works():
    mock = MyMock()
    model = MyModel()
    model.events.connect(mock.all)
    model.events.x.connect(mock.x)
    model.events.p.connect(mock.p)
    model.events.s.connect(mock.sub)
    model.sub.events.connect(mock.sub_all)
    model.sub.events.y.connect(mock.y)
    model.ls.events.connect(mock.ls_all)
    model.ls[0].events.connect(mock.ls_sub)

    model.x = 1
    model.sub.y = 1
    model.ls[0].y = 1
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.sub_calls == 0
    assert mock.sub_all_calls == 1
    assert mock.y_calls == 1
    assert mock.ls_all_calls == 1
    assert mock.ls_sub_calls == 1

    _disconnect_all_events(model, mock)

    # everyting should be disconnected
    model.x = 2
    model.sub.y = 2
    model.ls[0].y = 2
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.sub_calls == 0
    assert mock.sub_all_calls == 1
    assert mock.y_calls == 1
    assert mock.ls_all_calls == 1
    assert mock.ls_sub_calls == 1

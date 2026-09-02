import gc
from dataclasses import dataclass
from unittest.mock import Mock

from napari.utils.events import EventedModel, disconnect_events
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


@dataclass
class MyMock:
    # mock does not work here cause it creates methods on the fly and
    # they won't be detected by our machinery
    all_calls = 0
    x_calls = 0
    p_calls = 0
    s_calls = 0
    s_all_calls = 0
    y_calls = 0

    def all(self):
        self.all_calls += 1

    def x(self):
        self.x_calls += 1

    def p(self):
        self.p_calls += 1

    def s(self):
        self.s_calls += 1

    def s_all(self):
        self.s_all_calls += 1

    def y(self):
        self.y_calls += 1


class SubModel(EventedModel):
    y: int


class MyModel(EventedModel):
    x: int
    s: SubModel

    @property
    def p(self):
        return self.x + 1


def test_disconnect_events_works():
    mock = MyMock()
    model = MyModel(x=0, s={'y': 0})
    model.events.connect(mock.all)
    model.events.x.connect(mock.x)
    model.events.s.connect(mock.s)
    model.events.p.connect(mock.p)
    model.s.events.connect(mock.s_all)
    model.s.events.y.connect(mock.y)

    model.x = 1
    model.s.y = 1
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.s_calls == 0
    assert mock.s_all_calls == 1
    assert mock.y_calls == 1

    # only events from one emittergroup are disconnected
    disconnect_events(model.events, mock)

    model.x = 2
    model.s.y = 2
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.s_calls == 0
    assert mock.s_all_calls == 2
    assert mock.y_calls == 2

    disconnect_events(model.s.events, mock)

    model.x = 3
    model.s.y = 3
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.s_calls == 0
    assert mock.s_all_calls == 2
    assert mock.y_calls == 2


def test_disconnect_all_events_works():
    mock = MyMock()
    model = MyModel(x=0, s={'y': 0})
    model.events.connect(mock.all)
    model.events.x.connect(mock.x)
    model.events.s.connect(mock.s)
    model.events.p.connect(mock.p)
    model.s.events.connect(mock.s_all)
    model.s.events.y.connect(mock.y)

    model.x = 1
    model.s.y = 1
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.s_calls == 0
    assert mock.s_all_calls == 1
    assert mock.y_calls == 1

    _disconnect_all_events(model, mock)

    # everyting should be disconnected
    model.x = 2
    model.s.y = 2
    assert mock.all_calls == 1
    assert mock.x_calls == 1
    assert mock.p_calls == 1
    assert mock.s_calls == 0
    assert mock.s_all_calls == 1
    assert mock.y_calls == 1

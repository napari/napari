import gc
from unittest.mock import Mock

from napari.utils.events.event import Event, EventEmitter
from napari.utils.events.event_utils import (
    connect_no_arg,
    connect_setattr,
    connect_setattr_value,
)


def test_connect_no_arg():
    mock = Mock(["meth"])
    emiter = EventEmitter()
    connect_no_arg(emiter, mock, "meth")
    emiter(type_name="a", value=1)
    mock.meth.assert_called_once_with()
    assert len(emiter.callbacks) == 1
    del mock
    gc.collect()
    assert len(emiter.callbacks) == 1
    emiter(type_name="a", value=1)
    assert len(emiter.callbacks) == 0


def test_connect_setattr_value():
    mock = Mock()
    emiter = EventEmitter()
    connect_setattr_value(emiter, mock, "meth")
    emiter(type_name="a", value=1)
    assert mock.meth == 1
    assert len(emiter.callbacks) == 1
    del mock
    gc.collect()
    assert len(emiter.callbacks) == 1
    emiter(type_name="a", value=1)
    assert len(emiter.callbacks) == 0


def test_connect_setattr():
    mock = Mock()
    emiter = EventEmitter()
    connect_setattr(emiter, mock, "meth")
    emiter(type_name="a", value=1)
    assert isinstance(mock.meth, Event)
    assert mock.meth.value == 1
    assert len(emiter.callbacks) == 1
    del mock
    gc.collect()
    assert len(emiter.callbacks) == 1
    emiter(type_name="a", value=1)
    assert len(emiter.callbacks) == 0

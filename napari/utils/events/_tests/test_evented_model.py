from typing import ClassVar
from unittest.mock import Mock

from napari.utils.events import EmitterGroup, EventedModel


def test_creating_empty_evented_model():
    """Test creating an empty evented pydantic model."""
    model = EventedModel()
    assert model is not None
    assert model.events is not None


def test_evented_model():
    """Test creating an evented pydantic model."""

    class User(EventedModel):
        """Test an evented model.

        Parameters
        ----------
        id : int
            User id.
        name : str, optional
            User name.
        """

        id: int
        name: str = 'Alex'
        age: ClassVar[int] = 100

    user = User(id=0)
    # test basic functionality
    assert user.id == 0
    assert user.name == 'Alex'

    user.id = 2
    assert user.id == 2

    # test event system
    assert isinstance(user.events, EmitterGroup)
    assert 'id' in user.events
    assert 'name' in user.events

    # ClassVars are excluded from events
    assert 'age' not in user.events
    # mocking EventEmitters to spy on events
    user.events.id = Mock(user.events.id)
    user.events.name = Mock(user.events.name)
    # setting an attribute should, by default, emit an event with the value
    user.id = 4
    user.events.id.assert_called_with(value=4)
    user.events.name.assert_not_called()
    # and event should only be emitted when the value has changed.
    user.events.id.reset_mock()
    user.id = 4
    user.events.id.assert_not_called()
    user.events.name.assert_not_called()

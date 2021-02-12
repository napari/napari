from typing import ClassVar
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
from pydantic import Field

from napari.utils.events import EmitterGroup, EventedModel
from napari.utils.events.custom_types import Array


def test_creating_empty_evented_model():
    """Test creating an empty evented pydantic model."""
    model = EventedModel()
    assert model is not None
    assert model.events is not None


def test_evented_model():
    """Test creating an evented pydantic model."""

    class User(EventedModel):
        """Demo evented model.

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


def test_evented_model_with_array():
    """Test creating an evented pydantic model with an array."""

    def make_array():
        return np.array([4, 3])

    class Model(EventedModel):
        """Demo evented model."""

        int_values: Array[int]
        any_values: Array
        shaped1_values: Array[float, (-1,)]
        shaped2_values: Array[int, (1, 2)] = Field(default_factory=make_array)
        shaped3_values: Array[float, (4, -1)]
        shaped4_values: Array[float, (-1, 4)]

    model = Model(
        int_values=[1, 2.2, 3],
        any_values=[1, 2.2],
        shaped1_values=np.array([1.1, 2.0]),
        shaped3_values=np.array([1.1, 2.0, 2.0, 3.0]),
        shaped4_values=np.array([1.1, 2.0, 2.0, 3.0]),
    )
    # test basic functionality
    np.testing.assert_almost_equal(model.int_values, np.array([1, 2, 3]))
    np.testing.assert_almost_equal(model.any_values, np.array([1, 2.2]))
    np.testing.assert_almost_equal(model.shaped1_values, np.array([1.1, 2.0]))
    np.testing.assert_almost_equal(model.shaped2_values, np.array([[4, 3]]))
    np.testing.assert_almost_equal(
        model.shaped3_values, np.array([[1.1, 2.0, 2.0, 3.0]]).T
    )
    np.testing.assert_almost_equal(
        model.shaped4_values, np.array([[1.1, 2.0, 2.0, 3.0]])
    )

    # try changing shape to something impossible to correctly reshape
    with pytest.raises(ValueError):
        model.shaped2_values = [1]


def test_evented_model_array_updates():
    """Test updating an evented pydantic model with an array."""

    class Model(EventedModel):
        """Demo evented model."""

        values: Array[int]

    model = Model(values=[1, 2, 3])

    # Mock events
    model.events.values = Mock(model.events.values)

    np.testing.assert_almost_equal(model.values, np.array([1, 2, 3]))

    # Updating with new data
    model.values = [1, 2, 4]
    assert model.events.values.call_count == 1
    np.testing.assert_almost_equal(
        model.events.values.call_args[1]['value'], np.array([1, 2, 4])
    )
    model.events.values.reset_mock()

    # Updating with same data, no event should be emitted
    model.values = [1, 2, 4]
    model.events.values.assert_not_called()


def test_evented_model_array_equality():
    """Test checking equality with an evented model with custom array."""

    class Model(EventedModel):
        """Demo evented model."""

        values: Array[int]

    model1 = Model(values=[1, 2, 3])
    model2 = Model(values=[1, 5, 6])

    assert model1 == model1
    assert model1 != model2

    model2.values = [1, 2, 3]
    assert model1 == model2


def test_evented_model_np_array_equality():
    """Test checking equality with an evented model with direct numpy."""

    class Model(EventedModel):
        values: np.ndarray

    model1 = Model(values=np.array([1, 2, 3]))
    model2 = Model(values=np.array([1, 5, 6]))

    assert model1 == model1
    assert model1 != model2

    model2.values = np.array([1, 2, 3])
    assert model1 == model2


def test_evented_model_da_array_equality():
    """Test checking equality with an evented model with direct dask."""

    class Model(EventedModel):
        values: da.Array

    r = da.ones((64, 64))
    model1 = Model(values=r)
    model2 = Model(values=da.ones((64, 64)))

    assert model1 == model1
    # dask arrays will only evaluate as equal if they are the same object.
    assert model1 != model2

    model2.values = r
    assert model1 == model2


def test_values_updated():
    class User(EventedModel):
        """Demo evented model.

        Parameters
        ----------
        id : int
            User id.
        name : str, optional
            User name.
        """

        id: int
        name: str = 'A'
        age: ClassVar[int] = 100

    user1 = User(id=0)
    user2 = User(id=1, name='K')

    # Add mocks
    user1_events = Mock(user1.events)
    user1.events.connect(user1_events)
    user1.events.id = Mock(user1.events.id)
    user2.events.id = Mock(user2.events.id)

    # Check user1 and user2 dicts
    assert user1.dict() == {'id': 0, 'name': 'A'}
    assert user2.dict() == {'id': 1, 'name': 'K'}

    # Update user1 from user2
    user1.update(user2)
    assert user1.dict() == {'id': 1, 'name': 'K'}

    user1.events.id.assert_called_with(value=1)
    user2.events.id.assert_not_called()
    assert user1_events.call_count == 1
    user1.events.id.reset_mock()
    user2.events.id.reset_mock()
    user1_events.reset_mock()

    # Update user1 from user2 again, no event emission expected
    user1.update(user2)
    assert user1.dict() == {'id': 1, 'name': 'K'}

    user1.events.id.assert_not_called()
    user2.events.id.assert_not_called()
    assert user1_events.call_count == 0

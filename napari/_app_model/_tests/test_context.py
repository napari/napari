from unittest.mock import Mock

import pytest

from napari._app_model.context._context import ContextMapping


def test_simple_mapping():
    data = {'a': 1, 'b': 2}
    mapping = ContextMapping(data)
    assert mapping['a'] == 1
    assert mapping['b'] == 2
    assert 'a' in mapping
    assert list(mapping) == ['a', 'b']
    assert len(mapping) == 2


def test_missed_key():
    data = {'a': 1, 'b': 2}
    mapping = ContextMapping(data)
    with pytest.raises(KeyError):
        mapping['c']


def test_callable_value():
    data = {'a': 1, 'b': Mock(return_value=2)}
    mapping = ContextMapping(data)
    assert mapping['a'] == 1
    assert mapping['b'] == 2
    assert mapping['b'] == 2  # it is important to use [] twice in this test
    data['b'].assert_called_once()

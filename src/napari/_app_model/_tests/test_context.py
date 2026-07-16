from unittest.mock import Mock

import pytest

from napari._app_model.context._context import (
    ContextMapping,
    create_context,
    get_context,
)


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


def test_context_integration():
    obj = {1, 2, 3}
    ctx = create_context(obj)
    ctx['a'] = 1
    ctx['b'] = Mock(return_value=2)

    assert isinstance(get_context(obj), ContextMapping)
    mapping = get_context(obj)
    assert mapping['a'] == 1
    assert mapping['b'] == 2
    assert mapping['b'] == 2  # it is important to use [] twice in this test
    ctx['b'].assert_called_once()

    mapping2 = get_context(obj)
    assert mapping2['b'] == 2
    assert ctx['b'].call_count == 2
    assert mapping2['b'] == 2
    assert ctx['b'].call_count == 2

from unittest.mock import Mock, call

import pytest

from napari.utils.events import EventedSet


@pytest.fixture
def regular_set():
    return set(range(5))


@pytest.fixture
def test_set(request, regular_set):
    test_set = EventedSet(regular_set)
    test_set.events = Mock(wraps=test_set.events)
    return test_set


@pytest.mark.parametrize(
    'meth',
    [
        # METHOD, ARGS, EXPECTED EVENTS
        # primary interface
        ('add', 2, []),
        ('add', 10, [call.changed(added={10}, removed={})]),
        ('discard', 2, [call.changed(added={}, removed={2})]),
        ('remove', 2, [call.changed(added={}, removed={2})]),
        ('discard', 10, []),
        # parity with set
        ('update', {3, 4, 5, 6}, [call.changed(added={5, 6}, removed={})]),
        (
            'difference_update',
            {3, 4, 5, 6},
            [call.changed(added={}, removed={3, 4})],
        ),
        (
            'intersection_update',
            {3, 4, 5, 6},
            [call.changed(added={}, removed={0, 1, 2})],
        ),
        (
            'symmetric_difference_update',
            {3, 4, 5, 6},
            [call.changed(added={5, 6}, removed={3, 4})],
        ),
    ],
    ids=lambda x: x[0],
)
def test_set_interface_parity(test_set, regular_set, meth):
    method_name, arg, expected = meth
    test_set_method = getattr(test_set, method_name)
    assert tuple(test_set) == tuple(regular_set)

    regular_set_method = getattr(regular_set, method_name)
    assert test_set_method(arg) == regular_set_method(arg)
    assert tuple(test_set) == tuple(regular_set)

    assert test_set.events.mock_calls == expected


def test_set_pop():
    test_set = EventedSet(range(3))
    test_set.events = Mock(wraps=test_set.events)
    test_set.pop()
    assert len(test_set.events.changed.call_args_list) == 1
    test_set.pop()
    assert len(test_set.events.changed.call_args_list) == 2
    test_set.pop()
    assert len(test_set.events.changed.call_args_list) == 3
    with pytest.raises(KeyError):
        test_set.pop()
    with pytest.raises(KeyError):
        test_set.remove(34)


def test_set_clear(test_set):
    assert test_set.events.mock_calls == []
    test_set.clear()
    assert test_set.events.mock_calls == [
        call.changed(added={}, removed={0, 1, 2, 3, 4})
    ]


@pytest.mark.parametrize(
    'meth',
    [
        ('difference', {3, 4, 5, 6}),
        ('intersection', {3, 4, 5, 6}),
        ('issubset', {3, 4}),
        ('issubset', {3, 4, 5, 6}),
        ('issubset', {1, 2, 3, 4, 5, 6}),
        ('issuperset', {3, 4}),
        ('issuperset', {3, 4, 5, 6}),
        ('issuperset', {1, 2, 3, 4, 5, 6}),
        ('symmetric_difference', {3, 4, 5, 6}),
        ('union', {3, 4, 5, 6}),
    ],
)
def test_set_new_objects(test_set, regular_set, meth):
    method_name, arg = meth
    test_set_method = getattr(test_set, method_name)
    assert tuple(test_set) == tuple(regular_set)

    regular_set_method = getattr(regular_set, method_name)
    result = test_set_method(arg)
    assert result == regular_set_method(arg)
    assert isinstance(result, (EventedSet, bool))
    assert result is not test_set

    assert test_set.events.mock_calls == []

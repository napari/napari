from napari.utils.events.containers import EventedList, NestableEventedList
from collections.abc import MutableSequence
import pytest


@pytest.fixture
def regular_list():
    return list(range(5))


def flatten(_lst):
    return (
        flatten(_lst[0]) + (flatten(_lst[1:]) if len(_lst) > 1 else [])
        if isinstance(_lst, MutableSequence)
        else [_lst]
    )


@pytest.fixture(params=[EventedList, NestableEventedList])
def test_list(request, regular_list):
    test_list = request.param(regular_list)
    test_list._events = []
    test_list.events.connect(test_list._events.append)
    return test_list


@pytest.mark.parametrize(
    'meth',
    [
        # method, args, expected events
        ('append', (3,), ('inserting', 'inserted')),
        ('clear', (), ('removing', 'removed') * 5),
        ('count', (3,), ()),
        ('extend', ([7, 8, 9],), ('inserting', 'inserted') * 3),
        ('index', (3,), ()),
        ('insert', (2, 10), ('inserting', 'inserted')),
        ('pop', (-2,), ('removing', 'removed')),
        ('remove', (3,), ('removing', 'removed')),
        ('reverse', (), ('reordered',)),
        # sort?
    ],
    ids=lambda x: x[0],
)
def test_list_api_parity(test_list, regular_list, meth):
    method_name, args, expected_events = meth
    test_list_method = getattr(test_list, method_name)
    regular_list_method = getattr(regular_list, method_name)
    assert tuple(test_list) == tuple(regular_list)
    assert test_list_method(*args) == regular_list_method(*args)
    assert tuple(test_list) == tuple(regular_list)
    assert tuple(e.type for e in test_list._events) == expected_events


def test_copy(test_list, regular_list):
    new_test = test_list.copy()
    new_reg = regular_list.copy()
    assert new_test != test_list
    assert id(new_test) != id(test_list)
    assert tuple(new_test) == tuple(test_list) == tuple(new_reg)
    assert not test_list._events


NEST = [0, [10, [110, [1110, 1111, 1112], 112], 12], 2]


def test_nested_indexing():
    """test that we can index a nested list with nl[1, 2, 3] syntax."""
    ne_list = NestableEventedList(NEST)
    indices = [tuple(int(x) for x in str(n)) for n in flatten(NEST)]
    for index in indices:
        assert ne_list[index] == int("".join(map(str, index)))


# indices in NEST that are actually lists
@pytest.fixture(params=[(), (1,), (1, 1), (1, 1, 1)], ids=lambda x: str(x))
def group_index(request, regular_list):
    return request.param


@pytest.mark.parametrize(
    'meth',
    [
        # method, args, expected events
        ('append', (3,), ('inserting', 'inserted')),
        ('clear', (), ('removing', 'removed') * 3),
        ('count', (110,), ()),
        ('extend', ([7, 8, 9],), ('inserting', 'inserted') * 3),
        ('index', (110,), ()),
        ('insert', (0, 10), ('inserting', 'inserted')),
        ('pop', (-1,), ('removing', 'removed')),
    ],
    ids=lambda x: x[0],
)
def test_nested_events(meth, group_index):
    ne_list = NestableEventedList(NEST)
    ne_list._events = []
    ne_list.events.connect(ne_list._events.append)

    method_name, args, expected_events = meth
    method = getattr(ne_list[group_index], method_name)
    if method_name == 'index' and group_index != (1, 1):
        # the expected value only occurs in index (1, 1)
        with pytest.raises(ValueError):
            method(*args)
    else:
        # make sure we can call the method without error
        method(*args)

    # make sure the correct event type and number was emitted
    assert tuple(e.type for e in ne_list._events) == expected_events

    # IMPORTANT: make sure that all of the events emitted by the root level
    # NestableEventedList show the nested (tuple form) index of the object
    # that emitted the original event
    for event in ne_list._events:
        if isinstance(group_index, int):
            group_index = (group_index,)
        if group_index == ():
            # in the root group, the index will be an int relative to root
            assert isinstance(event.index, int)
        else:
            assert event.index[:-1] == group_index

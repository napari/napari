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
        # METHOD, ARGS, EXPECTED EVENTS
        # primary interface
        ('insert', (2, 10), ('inserting', 'inserted')),  # create
        ('__getitem__', (2,), ()),  # read
        ('__setitem__', (2, 3), ('changed',)),  # update
        ('__setitem__', (slice(2), [1, 2]), ('changed',)),  # update slice
        ('__delitem__', (2,), ('removing', 'removed')),  # delete
        ('__delitem__', (slice(2),), ('removing', 'removed') * 2,),
        ('__delitem__', (slice(-3),), ('removing', 'removed') * 2,),
        ('__delitem__', (slice(-2, None),), ('removing', 'removed') * 2,),
        # inherited interface
        ('append', (3,), ('inserting', 'inserted')),
        ('clear', (), ('removing', 'removed') * 5),
        ('count', (3,), ()),
        ('extend', ([7, 8, 9],), ('inserting', 'inserted') * 3),
        ('index', (3,), ()),
        ('pop', (-2,), ('removing', 'removed')),
        ('remove', (3,), ('removing', 'removed')),
        ('reverse', (), ('reordered',)),
        # sort?
    ],
    ids=lambda x: x[0],
)
def test_list_interface_parity(test_list, regular_list, meth):
    method_name, args, expected_events = meth
    test_list_method = getattr(test_list, method_name)
    regular_list_method = getattr(regular_list, method_name)
    assert tuple(test_list) == tuple(regular_list)
    assert test_list_method(*args) == regular_list_method(*args)
    assert tuple(test_list) == tuple(regular_list)
    assert tuple(e.type for e in test_list._events) == expected_events


def test_list_interface_exceptions(test_list):
    bad_index = {'a': 'dict'}
    with pytest.raises(TypeError):
        test_list[bad_index]

    with pytest.raises(TypeError):
        test_list[bad_index] = 1

    with pytest.raises(TypeError):
        del test_list[bad_index]

    with pytest.raises(TypeError):
        test_list.insert([bad_index], 0)


def test_copy(test_list, regular_list):
    """Copying an evented list should return a same-class evented list."""
    new_test = test_list.copy()
    new_reg = regular_list.copy()
    assert id(new_test) != id(test_list)
    assert new_test == test_list
    assert tuple(new_test) == tuple(test_list) == tuple(new_reg)
    assert not test_list._events


def test_move(test_list):
    """Copying an evented list should return a same-class evented list."""
    before = tuple(test_list)
    assert before == (0, 1, 2, 3, 4)  # from fixture
    test_list.move(
        0, 3
    )  # pop the object at 0 and insert at current position 3
    expectation = (1, 2, 0, 3, 4)
    assert tuple(test_list) != before
    assert tuple(test_list) == expectation

    # move the other way
    before = tuple(test_list)
    test_list.move(
        3, 0
    )  # pop the object at 3 and insert at current position 0
    expectation = (0, 1, 2, 3, 4)


def test_slice(test_list, regular_list):
    """Slicing an evented list should return a same-class evented list."""
    test_slice = test_list[1:3]
    regular_slice = regular_list[1:3]
    assert tuple(test_slice) == tuple(regular_slice)
    assert isinstance(test_slice, test_list.__class__)


NEST = [0, [10, [110, [1110, 1111, 1112], 112], 12], 2]


def test_nested_indexing():
    """test that we can index a nested list with nl[1, 2, 3] syntax."""
    ne_list = NestableEventedList(NEST)
    indices = [tuple(int(x) for x in str(n)) for n in flatten(NEST)]
    for index in indices:
        assert ne_list[index] == int("".join(map(str, index)))


# indices in NEST that are themselves lists
@pytest.mark.parametrize(
    'group_index', [(), (1,), (1, 1), (1, 1, 1)], ids=lambda x: str(x)
)
@pytest.mark.parametrize(
    'meth',
    [
        # METHOD, ARGS, EXPECTED EVENTS
        # primary interface
        ('insert', (0, 10), ('inserting', 'inserted')),
        ('__getitem__', (2,), ()),  # read
        ('__setitem__', (2, 3), ('changed',)),  # update
        ('__delitem__', ((),), ('removing', 'removed')),  # delete
        ('__delitem__', ((1,),), ('removing', 'removed')),  # delete
        ('__delitem__', (2,), ('removing', 'removed')),  # delete
        ('__delitem__', (slice(2),), ('removing', 'removed') * 2,),
        ('__delitem__', (slice(-1),), ('removing', 'removed') * 2,),
        ('__delitem__', (slice(-2, None),), ('removing', 'removed') * 2,),
        # inherited interface
        ('append', (3,), ('inserting', 'inserted')),
        ('clear', (), ('removing', 'removed') * 3),
        ('count', (110,), ()),
        ('extend', ([7, 8, 9],), ('inserting', 'inserted') * 3),
        ('index', (110,), ()),
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

        ('__setitem__', (slice(2), [1, 2]), ('changed',)),  # update slice


def test_setting_nested_slice():
    ne_list = NestableEventedList(NEST)
    ne_list[(1, 1, 1, slice(2))] = [9, 10]
    assert tuple(ne_list[1, 1, 1]) == (9, 10, 1112)


@pytest.mark.parametrize(
    'param',
    [
        # NEST = [0, [10, [110, [1110, 1111, 1112], 112], 12], 2]
        [
            ((1, 0), (1, 1, 1, 0), (1, 2)),
            (),
            [0, [[110, [1111, 1112], 112]], 2, 10, 1110, 12],
        ],
        [
            ((1, 0), (1, 1, 1), (1, 2)),
            (1, -2),
            [0, [[110, 112], 10, [1110, 1111, 1112], 12], 2],
        ],
        [
            ((1, 0), (1, -2),),
            (),
            [0, [[110, [1110, 1111, 1112], 112]], 2, 12, 10],
        ],
    ],
    ids=lambda x: str(x),
)
def test_nested_move_multiple(param):
    source, dest, expectation = param
    ne_list = NestableEventedList(NEST)
    ne_list._events = []
    ne_list.events.connect(ne_list._events.append)
    ne_list.move_multiple(source, dest)
    assert tuple(flatten((ne_list))) == tuple(flatten(expectation))

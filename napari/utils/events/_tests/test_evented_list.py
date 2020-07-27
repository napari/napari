from collections.abc import MutableSequence
from unittest.mock import Mock

import pytest

from napari.utils.events import (
    EmitterGroup,
    EventedList,
    NestableEventedList,
    TypedEventedList,
    TypedNestableEventedList,
)


@pytest.fixture
def regular_list():
    return list(range(5))


def flatten(_lst):
    return (
        flatten(_lst[0]) + (flatten(_lst[1:]) if len(_lst) > 1 else [])
        if isinstance(_lst, MutableSequence)
        else [_lst]
    )


@pytest.fixture(
    params=[
        EventedList,
        NestableEventedList,
        TypedEventedList,
        TypedNestableEventedList,
    ]
)
def test_list(request, regular_list):
    test_list = request.param(regular_list)
    test_list.events = Mock(wraps=test_list.events)
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
        ('__setitem__', (slice(2, 2), [1, 2]), ('changed',)),  # update slice
        ('__delitem__', (2,), ('removing', 'removed')),  # delete
        ('__delitem__', (slice(2),), ('removing', 'removed') * 2,),
        ('__delitem__', (slice(0, 0),), ('removing', 'removed')),
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
        ('__add__', ([7, 8, 9],), ()),
        ('__iadd__', ([7, 9],), ('inserting', 'inserted') * 2),
        ('__radd__', ([7, 9],), ('inserting', 'inserted') * 2),
        # sort?
    ],
    ids=lambda x: x[0],
)
def test_list_interface_parity(test_list, regular_list, meth):
    method_name, args, expected = meth
    test_list_method = getattr(test_list, method_name)
    assert tuple(test_list) == tuple(regular_list)
    if hasattr(regular_list, method_name):
        regular_list_method = getattr(regular_list, method_name)
        assert test_list_method(*args) == regular_list_method(*args)
        assert tuple(test_list) == tuple(regular_list)
    else:
        test_list_method(*args)  # smoke test

    for call, expect in zip(test_list.events.call_args_list, expected):
        event = call.args[0]
        assert event.type == expect


def test_hash(test_list):
    assert id(test_list) == hash(test_list)


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
    test_list.events.assert_not_called()


def test_move(test_list):
    """Test the that we can move objects with the move method"""
    before = tuple(test_list)
    assert before == (0, 1, 2, 3, 4)  # from fixture
    # pop the object at 0 and insert at current position 3
    test_list.move(0, 3)
    expectation = (1, 2, 0, 3, 4)
    assert tuple(test_list) != before
    assert tuple(test_list) == expectation

    # move the other way
    before = tuple(test_list)
    # pop the object at 3 and insert at current position 0
    test_list.move(3, 0)
    expectation = (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    'sources,dest,expectation',
    [
        ((2,), 0, [2, 0, 1, 3, 4, 5, 6, 7]),  # move single item
        ([0, 2, 3], 6, [1, 4, 5, 0, 2, 3, 6, 7]),  # move back
        ([4, 7], 1, [0, 4, 7, 1, 2, 3, 5, 6]),  # move forward
        ([0, 5, 6], 3, [1, 2, 0, 5, 6, 3, 4, 7]),  # move in between
        ([slice(None, 3)], 6, [3, 4, 5, 0, 1, 2, 6, 7]),  # move slice back
        ([slice(5, 8)], 2, [0, 1, 5, 6, 7, 2, 3, 4]),  # move slice forward
        ([slice(1, 8, 2)], 3, [0, 2, 1, 3, 5, 7, 4, 6]),  # move slice between
        ([slice(None, 8, 3)], 4, [1, 2, 0, 3, 6, 4, 5, 7]),  # again
    ],
)
def test_move_multiple(sources, dest, expectation):
    """Test the that we can move objects with the move method"""
    el = EventedList(range(8))
    el.events = Mock(wraps=el.events)
    assert el == [0, 1, 2, 3, 4, 5, 6, 7]

    el.move_multiple(sources, dest)
    assert el == expectation
    el.events.moving.assert_called_once()
    el.events.moved.assert_called_once()
    el.events.reordered.assert_called_with(value=expectation)


def test_move_multiple_mimics_slice_reorder():
    """Test the that move_multiple provides the same result as slice insertion.
    """
    data = list(range(8))
    el = EventedList(data)
    el.events = Mock(wraps=el.events)
    assert el == data
    new_order = [1, 5, 3, 4, 6, 7, 2, 0]
    # this syntax
    el.move_multiple(new_order, 0)
    # is the same as this syntax
    data[:] = [data[i] for i in new_order]
    assert el == new_order
    assert el == data
    el.events.moving.assert_called_with(index=new_order, new_index=0)
    el.events.moved.assert_called_with(
        index=new_order, new_index=0, value=new_order,
    )
    el.events.reordered.assert_called_with(value=new_order)


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
        ('__add__', ([7, 8, 9],), ()),
        ('__iadd__', ([7, 9],), ('inserting', 'inserted') * 2),
    ],
    ids=lambda x: x[0],
)
def test_nested_events(meth, group_index):
    ne_list = NestableEventedList(NEST)
    ne_list.events = Mock(wraps=ne_list.events)

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
    for call, expected in zip(ne_list.events.call_args_list, expected_events):
        event = call.args[0]
        assert event.type == expected
        if group_index == ():
            # in the root group, the index will be an int relative to root
            assert isinstance(event.index, int)
        else:
            assert event.index[:-1] == group_index


def test_setting_nested_slice():
    ne_list = NestableEventedList(NEST)
    ne_list[(1, 1, 1, slice(2))] = [9, 10]
    assert tuple(ne_list[1, 1, 1]) == (9, 10, 1112)


@pytest.mark.parametrize(
    'param',
    [
        # indices           2       (2, 1)
        # original = [0, 1, [(2,0), [(2,1,0), (2,1,1)], (2,2)], 3, 4]
        [((2, 0), (2, 1, 1), (3,)), (-1), [0, 1, [[210], 22], 4, 20, 211, 3]],
        [((2, 0), (2, 1, 1), (3,)), (1), [0, 20, 211, 3, 1, [[210], 22], 4]],
    ],
)
def test_nested_move_multiple(param):
    """Test that moving multiple indices works and emits right events."""
    source, dest, expectation = param
    ne_list = NestableEventedList([0, 1, [20, [210, 211], 22], 3, 4])
    ne_list.events = Mock(wraps=ne_list.events)
    ne_list.move_multiple(source, dest)
    ne_list.events.reordered.assert_called_with(value=expectation)


def test_arbitrary_child_events():
    """Test that any object that supports the events protocol bubbles events.
    """

    class E:
        events = EmitterGroup(test=None)

    e_obj = E()
    root = NestableEventedList()
    b = NestableEventedList()

    observed = []
    root.events.connect(lambda e: observed.append(e))

    root.append(b)
    b.append(e_obj)
    e_obj.events.test(value="hi")

    obs = [(e.type, e.index, getattr(e, 'value', None)) for e in observed]
    expected = [
        ('inserting', 0, None),
        ('inserted', 0, b),
        ('inserting', (0, 0), None),
        ('inserted', (0, 0), e_obj),
        ('test', (0, 0), 'hi'),
    ]
    for o, e in zip(obs, expected):
        assert o == e


def test_evented_list_subclass():
    """Test that multiple inheritance maintains events from superclass."""

    class A:
        events = EmitterGroup(boom=None)

    class B(A, EventedList):
        pass

    lst = B([1, 2])
    assert hasattr(lst, 'events')
    assert 'boom' in lst.events.emitters
    assert lst == [1, 2]

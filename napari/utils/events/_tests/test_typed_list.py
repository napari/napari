from napari.utils.events.containers import (
    TypedEventedList,
    TypedList,
    TypedNestableEventedList,
)
import pytest


@pytest.fixture(params=[TypedList, TypedEventedList, TypedNestableEventedList])
def list_type(request):
    return request.param


def test_type_enforcement(list_type):
    """Test that TypedLists enforce type during mutation events."""
    a = list_type([1, 2, 3, 4], basetype=int)
    assert tuple(a) == (1, 2, 3, 4)
    with pytest.raises(TypeError):
        a.append("string")
    with pytest.raises(TypeError):
        a.insert(0, "string")
    with pytest.raises(TypeError):
        a[0] = "string"
    with pytest.raises(TypeError):
        a[0] = 1.23

    # also on instantiation
    with pytest.raises(TypeError):
        _ = list_type([1, 2, '3'], basetype=int)


def test_multitype_enforcement(list_type):
    """Test that basetype also accepts/enforces a sequence of types."""
    a = list_type([1, 2, 3, 4, 5.5], basetype=(int, float))
    assert tuple(a) == (1, 2, 3, 4, 5.5)
    with pytest.raises(TypeError):
        a.append("string")
    a.append(2)
    a.append(2.4)


def test_custom_lookup(list_type):
    class Custom:
        def __init__(self, name='', data=()):
            self.name = name
            self.data = data

    hi = Custom(name='hi')
    tup = Custom(data=(1, 2, 3))

    a = list_type(
        [Custom(), hi, Custom(), tup],
        basetype=Custom,
        lookup={str: lambda x: x.name, tuple: lambda x: x.data},
    )
    # index with integer as usual
    assert a[1].name == 'hi'
    assert a.index("hi") == 1

    # index with string also works
    assert a['hi'] == hi

    # index with tuple will use the `tuple` type lookup
    assert a[(1, 2, 3)].data == (1, 2, 3)
    assert a.index((1, 2, 3)) == 3
    assert a[(1, 2, 3)] == tup

    # index still works with start/stop arguments
    with pytest.raises(ValueError):
        assert a.index((1, 2, 3), stop=2)
    with pytest.raises(ValueError):
        assert a.index((1, 2, 3), start=-3, stop=-1)

    # contains works
    assert 'hi' in a
    assert 'asdfsad' not in a

    # deletion works
    del a['hi']
    assert hi not in a
    assert 'hi' not in a

    del a[0]
    repr(a)


def test_nested_type_enforcement():
    data = [1, 2, [3, 4, [5, 6]]]
    a = TypedNestableEventedList(data, basetype=int)
    assert a[2, 2, 1] == 6

    # first level
    with pytest.raises(TypeError):
        a.append("string")
    with pytest.raises(TypeError):
        a.insert(0, "string")
    with pytest.raises(TypeError):
        a[0] = "string"

    # deeply nested
    with pytest.raises(TypeError):
        a[2, 2].append("string")
    with pytest.raises(TypeError):
        a[2, 2].insert(0, "string")
    with pytest.raises(TypeError):
        a[2, 2, 0] = "string"

    # also works during instantiation
    with pytest.raises(TypeError):
        _ = TypedNestableEventedList([1, 1, ['string']], basetype=int)

    with pytest.raises(TypeError):
        _ = TypedNestableEventedList([1, 2, [3, ['string']]], basetype=int)


@pytest.mark.xfail(reason="Need to enable custom indexing on nestable lists")
def test_nested_custom_lookup():
    class Custom:
        def __init__(self, name=''):
            self.name = name

    c = Custom()
    c1 = Custom(name='c1')
    c2 = Custom(name='c2')
    c3 = Custom(name='c3')

    a = TypedNestableEventedList(
        [c, c1, [c2, [c3]]], basetype=Custom, lookup={str: lambda x: x.name},
    )
    # first level
    assert a[1].name == 'c1'  # index with integer as usual
    assert a.index("c1") == 1
    assert a['c1'] == c1  # index with string also works

    # second level
    assert a[2, 0].name == 'c2'
    assert a.index("c2") == (2, 0)
    assert a['c2'] == c2

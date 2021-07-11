import pytest

from napari.utils.events.containers import EventedDict, TypedMutableMapping


# this is a parametrized fixture, all tests using ``dict_type`` will be run
# once using each of the items in params
# https://docs.pytest.org/en/stable/fixture.html#parametrizing-fixtures
@pytest.fixture(params=[TypedMutableMapping, EventedDict])
def dict_type(request):
    return request.param


def test_type_enforcement(dict_type):
    """Test that TypedDicts enforce type during mutation events."""
    a = dict_type({"A": 1, "B": 3, "C": 5}, basetype=int)
    assert tuple(a.values()) == (1, 3, 5)
    with pytest.raises(TypeError):
        a["D"] = "string"
    with pytest.raises(TypeError):
        a.update({"E": 3.5})

    # also on instantiation
    with pytest.raises(TypeError):
        dict_type({"A": 1, "B": 3.3, "C": "5"}, basetype=int)


def test_multitype_enforcement(dict_type):
    """Test that basetype also accepts/enforces a sequence of types."""
    a = dict_type({"A": 1, "B": 3, "C": 5.5}, basetype=(int, float))
    assert tuple(a.values()) == (1, 3, 5.5)
    with pytest.raises(TypeError):
        a["D"] = "string"
    a["D"] = 2.4
    a.update({"E": 3.5})

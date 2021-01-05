import inspect
import operator
from dataclasses import InitVar, asdict, field
from functools import partial
from typing import ClassVar, List, Optional
from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
from typing_extensions import Annotated

from napari.layers.base._base_constants import Blending
from napari.layers.utils._text_constants import Anchor
from napari.utils.events import EmitterGroup
from napari.utils.events.dataclass import (
    Property,
    _type_to_compare,
    evented_dataclass,
    is_equal,
)


@pytest.mark.parametrize("props, events", [(1, 1), (0, 1), (0, 0), (1, 0)])
def test_dataclass_with_properties(props, events):
    """Test that the @dataclass decorator works.

    This test is a bit long... but parameters test all combinations of props
    and events to make sure they work alone as well as together.
    """

    @evented_dataclass(properties=props, events=events)
    class M:
        """Just a test.

        Parameters
        ----------
        a : int
            Description of parameter `a`.
        b : str, optional
            Description of parameter `b`. by default 'hi'
        c : list, optional
            Description of parameter `c`. by default empty.
        """

        a: int
        b: str = 'hi'
        c: List[int] = field(default_factory=list)
        d: ClassVar[int] = 1
        e: int = field(default=1, metadata={"events": False})

        def _on_b_set(self, value):
            # NB: if you want to set value again, you must check that it is
            # actually different from ``value``!
            if value != 'bossy':
                self.b = 'bossy'

        def _on_c_set(self, value):
            if value == [1, 2]:
                return True

    m = M(a=1)
    # basic functionality
    assert m.a == 1
    assert m.b == 'hi'
    assert m.c == []
    m.a = 7
    m.c.append(9)
    # nice function ... note the ClassVar is missing
    assert asdict(m) == {'a': 7, 'b': 'hi', 'c': [9], 'e': 1}

    assert isinstance(m.a, int)
    assert isinstance(m.b, str)
    if props:
        # The fields should have been converted to property descriptors
        assert isinstance(M.a, property)
        assert isinstance(M.b, property)
        # and their docstrings pulled from the class (numpy) docstring
        assert M.a.__doc__ == "Description of parameter `a`."
        assert M.b.__doc__ == "Description of parameter `b`. by default 'hi'"
        assert M.c.__doc__ == "Description of parameter `c`. by default empty."
    else:
        # otherwise fields should not be property descriptors
        assert not hasattr(M, 'a')

    if events:
        # an EmmiterGroup named `events` should have been added to the class.
        assert isinstance(m.events, EmitterGroup)
        assert 'a' in m.events
        assert 'b' in m.events
        # ClassVars and metadata={'events'=True} are excluded from events
        assert 'd' not in m.events
        assert 'e' not in m.events
        # mocking EventEmitters to spy on events
        m.events.a = Mock(m.events.a)
        m.events.b = Mock(m.events.b)
        m.events.c = Mock(m.events.c)
        # setting an attribute should, by default, emit an event with the value
        m.a = 4
        m.events.a.assert_called_with(value=4)
        # and event should only be emitted when the value has changed.
        m.events.a.reset_mock()
        m.a = 4
        m.events.a.assert_not_called()

        # test that our _on_b_set override worked, and emitted the right event
        m.b = 'howdie'
        assert m.b == 'bossy'
        m.events.b.assert_called_with(value='bossy')

        # test that _on_c_set prevented an event by returning True
        m.c = [1, 2]
        assert m.c == [1, 2]
        m.events.c.assert_not_called()

    else:
        assert not hasattr(m, 'events')


def test_dataclass_missing_vars_raises():
    @evented_dataclass(properties=True, events=False)
    class M:
        a: int
        b: list = field(default_factory=list)
        c: str = field(default='asdf')
        d: int = 9
        # ClassVars are ignored entirely by dataclasses
        e: ClassVar[int] = 1
        f: ClassVar[str]

    with pytest.raises(TypeError) as excinfo:
        _ = M()  # missing `a`
    assert "missing 1 required positional argument" in str(excinfo.value)
    assert M(1).a == 1
    m = M(a=2)
    assert m.a == 2
    assert m.b == []
    assert m.c == 'asdf'
    assert m.d == 9
    # Classvars and _private property names are left out of dict
    assert asdict(m) == {'a': 2, 'b': [], 'c': 'asdf', 'd': 9}
    # ClassVars must have a default value to be seen as attributes.
    assert m.e == 1
    assert M.e == 1
    # Otherwise they are just annotations
    assert not hasattr(m, 'f')
    assert not hasattr(M, 'f')
    assert 'f' in M.__annotations__


def test_dataclass_coerces_types():
    @evented_dataclass(properties=True, events=False)
    class M:
        x: int = 2
        anchor: Annotated[Anchor, str, Anchor] = Anchor.UPPER_LEFT
        # Property is an alias for Annotated, and provides stricter checking
        blending: Property[Blending, None, Blending] = Blending.OPAQUE

    m = M()
    m.anchor = 'center'
    assert isinstance(m._anchor, Anchor)
    assert isinstance(m.anchor, str)

    assert isinstance(m.blending, Blending)
    m.blending = 'additive'
    assert isinstance(m._blending, Blending)
    assert m.blending == Blending.ADDITIVE


def test_Property_validation():
    # must provide at least 2 arguments
    with pytest.raises(TypeError):
        _ = Property[int]
    # wrong syntax
    with pytest.raises(TypeError):
        _ = Property(int, None)
    # the getter/setter must be callable
    with pytest.raises(TypeError):
        _ = Property[int, 1]
    with pytest.raises(TypeError):
        _ = Property[int, None, 1]
    # there we go
    assert Property[int, int, None]


def test_exception_resets_value():
    @evented_dataclass(events=True, properties=False)
    class M:
        x: int = 2

        def _on_x_set(self, val):
            raise ValueError('no can do')

    m = M()
    with pytest.raises(ValueError) as exc:
        m.x = 5
    assert 'Error in M._on_x_set (value not set): no can do' in str(exc)
    assert m.x == 2


def test_event_inheritance():
    """Test that subclasses include events from the superclass."""

    @evented_dataclass(events=True, properties=False)
    class A:
        a: int = 4
        x: int = 2

    @evented_dataclass(events=True, properties=False)
    class B(A):
        a: int = 2
        z: int = 4

    b = B(1)
    assert asdict(b) == {'a': 1, 'x': 2, 'z': 4}
    assert set(b.events.emitters) == {'z', 'a', 'x'}
    for key in {'z', 'a', 'x'}:
        setattr(b.events, key, Mock(getattr(b.events, key)))
        setattr(b, key, 10)
        getattr(b.events, key).assert_called_with(value=10)


def test_event_partial_inheritance():
    """Test events only included from classes decorated with events=True."""

    @evented_dataclass(events=False, properties=False)
    class A:
        a: int = 4
        x: int = 2

    @evented_dataclass(events=True, properties=False)
    class B(A):
        a: int = 2
        z: int = 4

    assert set(B().events.emitters) == {'z', 'a'}

    @evented_dataclass(events=True, properties=False)
    class C:
        a: int = 4
        x: int = 2

    @evented_dataclass(events=False, properties=False)
    class D(C):
        a: int = 2
        z: int = 4

    assert set(D().events.emitters) == {'x', 'a'}


def test_dataclass_signature():
    @evented_dataclass(properties=True, events=True)
    class A:
        a: str
        b: int = 2
        # c: Property[Anchor, str, Anchor] = Anchor.CENTER

    assert str(inspect.signature(A)) == '(a: str, b: int = 2) -> None'


def test_values_updated():
    @evented_dataclass(properties=True, events=True)
    class A:
        a: str
        b: int = 2

    obj1 = A("a", 2)
    obj2 = A("b", 2)
    obj3 = A("a", 1)

    assert obj1.asdict() == {"a": "a", "b": 2}
    assert obj2.asdict() == {"a": "b", "b": 2}

    count = {"a": 0, "b": 0, "values_updated": 0}

    def count_calls(name, event):
        count[name] += 1

    obj2.events.a.connect(partial(count_calls, "a"))
    obj2.events.b.connect(partial(count_calls, "b"))
    obj2.events.connect(partial(count_calls, "values_updated"))

    obj2.update(obj1.asdict())

    assert obj2.asdict() == {"a": "a", "b": 2}
    assert count == {"a": 1, "b": 0, "values_updated": 1}

    count = {"a": 0, "b": 0, "values_updated": 0}
    obj2.update({"a": "c", "b": 3})
    assert count == {"a": 1, "b": 1, "values_updated": 1}

    count = {"a": 0, "b": 0, "values_updated": 0}
    obj2.update(obj3)
    assert count == {"a": 1, "b": 1, "values_updated": 1}

    count = {"a": 0, "b": 0, "values_updated": 0}
    obj2.update(obj2)
    assert count == {"a": 0, "b": 0, "values_updated": 0}


def test_is_equal_warnings():
    with pytest.warns(UserWarning, match="Comparison method failed*"):
        assert not is_equal(np.ones(2), np.ones(2))

    with pytest.warns(UserWarning, match="Comparison method failed*"):
        assert not is_equal(
            [np.ones(2), np.zeros(2)], [np.ones(2), np.zeros(2)]
        )

    with pytest.warns(UserWarning, match="Comparison method failed*"):
        assert not is_equal(
            {1: np.ones(2), 2: np.zeros(2)},
            {1: np.ones(2), 2: np.zeros(2)},
        )


def test_is_equal():
    assert is_equal(1, 1)
    assert is_equal(1, 1.0)
    assert not is_equal(1, 2)


def test_type_to_compare():
    assert _type_to_compare(int) is None
    assert _type_to_compare(Property[int, None, int]) is None
    assert _type_to_compare(np.ndarray) is np.array_equal
    assert (
        _type_to_compare(Property[np.ndarray, None, np.array])
        is np.array_equal
    )
    assert _type_to_compare(da.core.Array) is operator.is_
    assert (
        _type_to_compare(Property[da.core.Array, None, da.from_array])
        is operator.is_
    )


def test_values_updated_complex_type():
    @evented_dataclass(properties=True, events=True)
    class A:
        a: int
        b: np.ndarray
        c: da.core.Array

    da_array = da.from_array(np.zeros(2))
    obj1 = A(1, np.ones(2), da_array)
    count = {"a": 0, "b": 0, "c": 0}

    def count_calls(name, event):
        count[name] += 1

    obj1.events.a.connect(partial(count_calls, "a"))
    obj1.events.b.connect(partial(count_calls, "b"))
    obj1.events.c.connect(partial(count_calls, "c"))

    obj1.a = 1
    assert count == {"a": 0, "b": 0, "c": 0}
    obj1.a = 2
    assert count == {"a": 1, "b": 0, "c": 0}
    obj1.b = np.ones(2)
    assert count == {"a": 1, "b": 0, "c": 0}
    obj1.b = np.zeros(2)
    assert count == {"a": 1, "b": 1, "c": 0}
    obj1.c = da_array
    assert count == {"a": 1, "b": 1, "c": 0}
    obj1.c = da.from_array(np.zeros(2))
    assert count == {"a": 1, "b": 1, "c": 1}
    obj1.c = da.from_array(np.ones(2))
    assert count == {"a": 1, "b": 1, "c": 2}


def test_values_updated_array():
    @evented_dataclass(properties=True, events=True)
    class A:
        a: str
        b: int = 2

    obj1 = A("a", 2)
    count = {"b": 0}

    def count_calls(name, event):
        count[name] += 1

    obj1.events.b.connect(partial(count_calls, "b"))
    with pytest.warns(UserWarning, match="Comparison method failed*"):
        obj1.b = np.array([2, 2])
    assert count["b"] == 1


def test_init_var_warning():
    with pytest.warns(None) as record:

        @evented_dataclass
        class T:
            colors: InitVar[str] = 'black'

    assert len(record) == 0


def test_optional_numpy_warning():
    with pytest.warns(None) as record:

        @evented_dataclass
        class T:
            colors: Optional[np.ndarray]

        t = T(None)

        t.colors = np.arange(3)

    assert len(record) == 0

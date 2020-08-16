import inspect
from dataclasses import asdict, field
from typing import ClassVar, List
from unittest.mock import Mock

import pytest
from typing_extensions import Annotated

from napari.layers.base._base_constants import Blending
from napari.layers.utils._text_constants import Anchor
from napari.utils.events import EmitterGroup
from napari.utils.events.dataclass import Property, dataclass


@pytest.mark.parametrize("props, events", [(1, 1), (0, 1), (0, 0), (1, 0)])
def test_dataclass_with_properties(props, events):
    """Test that the @dataclass decorator works.

    This test is a bit long... but parameters test all combinations of props
    and events to make sure they work alone as well as together.
    """

    @dataclass(properties=props, events=events)
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
    @dataclass(properties=True)
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
    @dataclass(properties=True)
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
    @dataclass(events=True)
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

    @dataclass(events=True)
    class A:
        a: int = 4
        x: int = 2

    @dataclass(events=True)
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

    @dataclass
    class A:
        a: int = 4
        x: int = 2

    @dataclass(events=True)
    class B(A):
        a: int = 2
        z: int = 4

    assert set(B().events.emitters) == {'z', 'a'}

    @dataclass(events=True)
    class C:
        a: int = 4
        x: int = 2

    @dataclass
    class D(C):
        a: int = 2
        z: int = 4

    assert set(D().events.emitters) == {'x', 'a'}


def test_dataclass_signature():
    @dataclass(properties=True, events=True)
    class A:
        a: str
        b: int = 2
        # c: Property[Anchor, str, Anchor] = Anchor.CENTER

    assert str(inspect.signature(A)) == '(a: str, b: int = 2) -> None'

from dataclasses import asdict, field
from typing import List
from unittest.mock import Mock

import pytest

from napari.utils.dataclass import dataclass
from napari.utils.event import EmitterGroup


@pytest.mark.parametrize("props, events", [(1, 1), (0, 1), (0, 0), (1, 0)])
def test_dataclass_with_properties(props, events):
    """Test that the @dataclass decorator works.

    The parameters test all combinations of props and events to make sure they
    work alone as well as together.
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

        def _on_b_set(self, value):
            # NB: if you want to set value again, you must check that it is
            # actually different from ``value``!
            if value != 'bossy':
                self.b = 'bossy'

        def _on_c_set(self, value):
            if value == [1, 2]:
                return True

    # currently, properties will only be added to the class during post_init
    m = M(a=1)
    # basic functionality
    assert m.a == 1
    assert m.b == 'hi'
    assert m.c == []
    m.a = 7
    m.c.append(9)
    # nice function
    assert asdict(m) == {'a': 7, 'b': 'hi', 'c': [9]}

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
        # mocking EventEmitters to spy on events
        m.events.a = Mock(m.events.a)
        m.events.b = Mock(m.events.b)
        m.events.c = Mock(m.events.c)
        # setting an attribute should, by default, emit an event with the value
        m.a = 4
        m.events.a.assert_called_with(value=4)

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

    with pytest.raises(TypeError):
        _ = M()  # missing `a`
    assert M(1).a == 1
    assert M(a=1).a == 1

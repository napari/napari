from dataclasses import asdict, field
from typing import List
from unittest.mock import Mock

import pytest

from napari.utils.dataclass import dataclass
from napari.utils.event import EmitterGroup


@pytest.mark.parametrize("props, events", [(1, 1), (0, 1), (0, 0), (1, 0)])
def test_dataclass_with_properties(props, events):
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
        assert isinstance(M.a, property)
        assert isinstance(M.b, property)
        assert M.a.__doc__ == "Description of parameter `a`."
        assert M.b.__doc__ == "Description of parameter `b`. by default 'hi'"
        assert M.c.__doc__ == "Description of parameter `c`. by default empty."
    else:
        assert not hasattr(M, 'a')

    if events:
        assert isinstance(m.events, EmitterGroup)
        assert 'a' in m.events
        assert 'b' in m.events
        m.events.b = Mock(m.events.b)
        m.events.a = Mock(m.events.a)
        m.a = 4
        m.events.a.assert_called_with(value=4)
        m.b = 'howdie'
        m.events.b.assert_called_with(value='howdie')

from unittest.mock import Mock

import pytest

from napari._pydantic_compat import ValidationError
from napari.utils.events import EventedModel, Selection


def test_selection():
    class T(EventedModel):
        sel: Selection[int]

    t = T(sel=[])
    t.sel.events._current = Mock()
    assert not t.sel._current
    assert not t.sel
    t.sel.add(1)
    t.sel._current = 1
    t.sel.events._current.assert_called_once()

    assert 1 in t.sel
    assert t.sel._current == 1

    assert t.json() == r'{"sel": {"selection": [1], "_current": 1}}'
    assert T(sel={"selection": [1], "_current": 1}) == t

    t.sel.remove(1)
    assert not t.sel

    with pytest.raises(ValidationError):
        T(sel=['asdf'])

    with pytest.raises(ValidationError):
        T(sel={"selection": [1], "_current": 'asdf'})

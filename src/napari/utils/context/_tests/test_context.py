import gc
from unittest.mock import Mock

import pytest

from napari.utils.context import Context, create_context, get_context
from napari.utils.context._context import _OBJ_TO_CONTEXT, SettingsAwareContext


def test_create_context():
    """You can create a context for any object"""

    class T:
        ...

    t = T()
    tid = id(t)
    ctx = create_context(t)
    assert get_context(t) == ctx
    assert hash(ctx)  # hashable
    assert tid in _OBJ_TO_CONTEXT
    _OBJ_TO_CONTEXT.pop(tid)

    del t
    gc.collect()
    assert tid not in _OBJ_TO_CONTEXT

    # you can provide your own root, but it must be a context
    create_context(T(), root=Context())
    with pytest.raises(AssertionError):
        create_context(T(), root={})  # type: ignore


def test_create_and_get_scoped_contexts():
    """Test that objects created in the stack of another contexted object.

    likely the most common way that this API will be used:
    """
    before = len(_OBJ_TO_CONTEXT)

    class A:
        def __init__(self) -> None:
            create_context(self)
            self.b = B()

    class B:
        def __init__(self) -> None:
            create_context(self)

    obja = A()
    assert len(_OBJ_TO_CONTEXT) == before + 2
    ctxa = get_context(obja)
    ctxb = get_context(obja.b)
    assert ctxa and ctxb
    ctxa['hi'] = 'hi'
    assert ctxb['hi'] == 'hi'

    # keys get deleted on object deletion
    del obja
    gc.collect()
    assert len(_OBJ_TO_CONTEXT) == before


def test_context_events():
    """Changing context keys emits an event"""
    mock = Mock()
    root = Context()
    scoped = root.new_child()
    scoped.changed.connect(mock)  # connect the mock to the child

    root['a'] = 1
    # child re-emits parent events
    assert mock.call_args[0][0].value == {'a'}

    mock.reset_mock()
    scoped['b'] = 1
    # also emits own events
    assert mock.call_args[0][0].value == {'b'}

    mock.reset_mock()
    del scoped['b']
    assert mock.call_args[0][0].value == {'b'}

    # but parent does not emit child events
    mock.reset_mock()
    mock2 = Mock()
    root.changed.connect(mock2)
    scoped['c'] = 'c'
    mock.assert_called_once()
    mock2.assert_not_called()


def test_settings_context():
    """The root context is a SettingsAwareContext."""
    mock = Mock()
    root = SettingsAwareContext()
    root.changed.connect(mock)

    assert isinstance(root['settings.appearance'], dict)
    assert root['settings.appearance.theme'] == 'dark'
    from napari.settings import get_settings

    get_settings().appearance.theme = 'light'
    assert root['settings.appearance.theme'] == 'light'
    assert dict(root) == {}  # the context itself doesn't have the value
    event = mock.call_args_list[0][0][0]
    assert event.value == {'settings.appearance.theme'}

    # any changes made here can't affect the global settings...
    with pytest.raises(ValueError):
        root['settings.appearance.theme'] = 'dark'
    assert get_settings().appearance.theme == 'light'

    with pytest.raises(KeyError):
        # can't delete from settings here either
        del root['settings.appearance.theme']
    assert root['settings.appearance.theme'] == 'light'

    with pytest.raises(KeyError):
        # of course, keys not in settings should still raise key error
        root['not.there']

    # but can be added like any other context
    root['there'] = 1
    assert root['there'] == 1

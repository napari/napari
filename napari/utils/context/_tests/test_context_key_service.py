from unittest.mock import Mock

import pytest

from napari.utils.context._service import (
    Context,
    ContextKeyService,
    create_context,
    get_context,
)


def test_create_and_get_scoped_contexts():
    """Test that objects created in the stack of another contexted object.

    likely the most common way that this API will be used:
    """

    class A:
        def __init__(self) -> None:
            self.ctx = create_context(self)
            self.b = B()

    class B:
        def __init__(self) -> None:
            self.ctx = create_context(self)

    obj = A()
    ctxb = get_context(obj.b)
    assert ctxb
    assert ctxb._parent is get_context(obj) is obj.ctx


def test_context():
    """test Context object is a dict that inherits values from parent."""
    A_DICT = {'a': 0, 'b': 0, 'c': 0}
    B_DICT = {'a': 1, 'd': 1}
    C_DICT = {'a': 2, 'c': 2, 'e': 2}
    a = Context(A_DICT)
    b = a._create_child(B_DICT)
    c = b._create_child(C_DICT)

    assert isinstance(a, dict)

    # dict() returns the context-local dict
    assert dict(a) == A_DICT
    assert dict(b) == B_DICT
    assert dict(c) == C_DICT

    # .collect() also includes parents
    assert a.collect() == A_DICT
    assert b.collect() == {**A_DICT, **B_DICT}
    assert c.collect() == {**A_DICT, **B_DICT, **C_DICT}

    # get pulls from parents when missing
    assert a['b'] == b['b'] == c['b'] == 0
    assert a.get('b') == b.get('b') == c.get('b') == 0
    assert a.get('f') is b.get('f') is c.get('f') is None
    assert a.get('f', 4) == b.get('f', 4) == c.get('f', 4) == 4

    # raises key error when missing
    with pytest.raises(KeyError):
        c['f']

    # repr shows parent keys in a comment when applicable
    assert '#' not in repr(a)
    assert "# <{'b': 0, 'c': 0}>" in repr(b)
    assert "# <{'b': 0, 'd': 1}>" in repr(c)


def test_root_service():
    """Root service, behaves a lot like a dict (but really "manages" it.)"""
    root = ContextKeyService()
    assert root._context_id == 0

    assert dict(root) == {}
    _id = id(root._my_context)

    # set
    root['key'] = 1

    # get
    assert root['key'] == 1
    assert dict(root) == {'key': 1}
    assert len(root) == 1
    assert 'key' in root

    # update
    root['key'] = 2
    assert dict(root) == {'key': 2}

    # delete
    del root['key']
    assert dict(root) == {}
    assert id(root._my_context) == _id  # make sure we haven't changed the obj

    assert repr(root)


class T:
    ...


def test_scoped_service_inherits():
    """Most contexts will be scoped on an object.

    Keys set in the scope override the root keys, but missing keys are pulled
    from the root context.
    """
    t = T()
    root = ContextKeyService()
    scoped = root.create_scoped(t)
    repr(scoped)

    assert root.get_context(t) == get_context(t) == scoped

    with pytest.raises(RuntimeError):
        root.create_scoped(t)

    # add to the scoped dict ... the root is unaffected
    scoped['k'] = 0
    assert dict(root) == {}
    assert dict(scoped) == {'T': 'T:0', 'k': 0}

    assert repr(scoped)

    # add to the root dict ... the scoped dict is unaffected
    root['k'] = 1
    assert dict(root) == {'k': 1}
    assert dict(scoped) == {'T': 'T:0', 'k': 0}
    assert scoped['k'] == 0

    # delete the scoped key, now it inherits from the parent
    del scoped['k']
    assert dict(root) == {'k': 1}
    assert dict(scoped) == {'T': 'T:0', 'k': 1}
    assert scoped['k'] == 1

    # deleting again has no consequence
    del scoped['k']
    assert dict(scoped) == {'T': 'T:0', 'k': 1}

    # deleting from parent affects the scoped keys
    scoped['k2'] = 10
    del root['k']
    assert dict(root) == {}
    assert dict(scoped) == {'T': 'T:0', 'k2': 10}

    # make sure that deletion frees the object for rescoping
    del scoped
    import gc

    gc.collect()
    newscope = root.create_scoped(t)
    assert newscope is not None

    # make sure that deletion of the object removes the scoped context from root
    assert newscope._context_id in root._contexts
    del t
    assert newscope._context_id not in root._contexts
    del newscope


def test_service_events():
    """Changing context keys emits an event"""
    mock = Mock()
    root = ContextKeyService()
    scoped = root.create_scoped(T())
    scoped.context_changed.connect(mock)

    root['a'] = 1
    event = mock.call_args_list[0][0][0]
    assert event.value == ['a']
    mock.reset_mock()

    scoped['b'] = 1
    event = mock.call_args_list[0][0][0]
    assert event.value == ['b']


def test_settings_context():
    """The root context is a SettingsAwareContext."""
    mock = Mock()
    root = ContextKeyService()
    root.context_changed.connect(mock)

    assert root['settings.appearance.theme'] == 'dark'
    from napari.settings import get_settings

    get_settings().appearance.theme = 'light'
    assert root['settings.appearance.theme'] == 'light'
    assert dict(root) == {}  # the context itself doesn't have the value
    event = mock.call_args_list[0][0][0]
    assert event.value == ['settings.appearance.theme']

    # any changes made here do not affect the global settings...
    # this is just a context that pulls from settings when missing.
    root['settings.appearance.theme'] = 'dark'
    assert 'settings.appearance.theme' in root
    assert root['settings.appearance.theme'] == 'dark'
    assert get_settings().appearance.theme == 'light'

    # complex objects are cast to dict
    assert isinstance(root['settings.appearance'], dict)

    del root['settings.appearance.theme']
    assert 'settings.appearance.theme' not in root
    assert root['settings.appearance.theme'] == 'light'  # falls back again

import inspect

import pytest

from .. import keybindings
from ..keybindings import (
    bind_key,
    components_to_key_combo,
    KeymapMixin,
    InheritedKeymap,
    normalize_key_combo,
    parse_key_combo,
)


def test_parse_key_combo():
    assert parse_key_combo('X') == ('X', set())
    assert parse_key_combo('Control-X') == ('X', {'Control'})
    assert parse_key_combo('Control-Alt-Shift-Meta-X') == (
        'X',
        {'Control', 'Alt', 'Shift', 'Meta'},
    )


def test_components_to_key_combo():
    assert components_to_key_combo('X', []) == 'X'
    assert components_to_key_combo('X', ['Control']) == 'Control-X'

    # test consuming
    assert components_to_key_combo('X', []) == 'X'
    assert components_to_key_combo('X', ['Shift']) == 'Shift-X'
    assert components_to_key_combo('x', []) == 'X'

    assert components_to_key_combo('@', ['Shift']) == '@'
    assert (
        components_to_key_combo('2', ['Control', 'Shift']) == 'Control-Shift-2'
    )

    # test ordering
    assert (
        components_to_key_combo('2', ['Control', 'Alt', 'Shift', 'Meta'])
        == 'Control-Alt-Shift-Meta-2'
    )
    assert (
        components_to_key_combo('2', ['Alt', 'Shift', 'Control', 'Meta'])
        == 'Control-Alt-Shift-Meta-2'
    )


def test_normalize_key_combo():
    assert normalize_key_combo('x') == 'X'
    assert normalize_key_combo('Control-X') == 'Control-X'
    assert normalize_key_combo('Meta-Alt-X') == 'Alt-Meta-X'
    assert (
        normalize_key_combo('Shift-Alt-Control-Meta-2')
        == 'Control-Alt-Shift-Meta-2'
    )


def test_bind_key():
    kb = {}

    # bind
    def forty_two():
        return 42

    bind_key(kb, 'A', forty_two)
    assert kb == dict(A=forty_two)

    with pytest.raises(TypeError):  # must check for callable
        bind_key(kb, 'B', 'not a callable')

    # overwrite
    def spam():
        return 'SPAM'

    with pytest.raises(ValueError):
        bind_key(kb, 'A', spam)

    bind_key(kb, 'A', spam, overwrite=True)
    assert kb == dict(A=spam)

    # unbind
    bind_key(kb, 'A', None)
    assert kb == {}


def test_bind_key_decorator():
    kb = {}

    @bind_key(kb, 'A')
    def foo():
        ...

    assert kb == dict(A=foo)


def test_InheritedKeymap():
    d = {}
    k = InheritedKeymap(lambda: d, {'A': 0})
    assert k['A'] == 0

    d['B'] = 1
    assert 'B' in k
    assert k['B'] == 1

    assert k.copy()['A'] == 0
    assert k.copy()['B'] == 1

    k['B'] = 2
    assert k['B'] == 2

    assert k.pop('B') == 2
    assert k.data['B'] is None
    with pytest.raises(KeyError):
        k['B']


def test_KeymapMixin():
    class Foo(KeymapMixin):
        ...

    foo = Foo()

    foo.bind_key('A', lambda: 42)
    assert foo.keymap['A']() == 42

    @Foo.bind_key('B')
    def bar():
        return 'SPAM'

    assert Foo.class_keymap['B'] is bar
    assert foo.keymap['B'] is bar

    foo.bind_key('B', lambda: 'aliiiens', overwrite=True)
    assert Foo.class_keymap['B'] is bar
    assert foo.keymap['B']() == 'aliiiens'

    foo.bind_key('B', None)
    assert foo.keymap.data['B'] is None
    with pytest.raises(KeyError):
        foo.keymap['B']

    class Bar(Foo):
        ...

    assert Bar.class_keymap is not Foo.class_keymap

    class Baz(Foo):
        class_keymap = dict(A=42)

    assert Baz.class_keymap == dict(A=42)


def test_bind_key_doc():
    doc = inspect.getdoc(bind_key)
    doc = doc.split('Notes\n-----\n')[-1]

    assert doc == inspect.getdoc(keybindings)

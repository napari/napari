import pytest
from ..keybindings import (_bind_keys_normalize_input, _bind_keys_validate_pair,
                           _determine_binding_op, _expand_shorthand,
                           BindOperation, bind_key, bind_keys, bind_key_method,
                           bind_keys_method, components_to_seq,
                           normalize_key_sequence, parse_seq, rebind_key,
                           rebind_key_method, rebind_keys, unbind_key,
                           unbind_key_method, unbind_keys)


def test_expand_shorthand():
    assert _expand_shorthand('x') == 'x'
    assert _expand_shorthand('C-x') == 'Control-x'
    assert _expand_shorthand('C-M-S-s-x') == 'Control-Alt-Shift-Meta-x'


def test_parse_seq():
    assert parse_seq('x') == ('x', set())
    assert parse_seq('Control-x') == ('x', {'Control'})
    assert parse_seq('Control-Alt-Shift-Meta-x') \
        == ('x', {'Control', 'Alt', 'Shift', 'Meta'})


def test_components_to_seq():
    assert components_to_seq('x', []) == 'x'
    assert components_to_seq('x', ['Control']) == 'Control-x'

    # test consuming
    assert components_to_seq('x', []) == 'x'
    assert components_to_seq('x', ['Shift']) == 'X'
    assert components_to_seq('X', []) == 'x'

    assert components_to_seq('@', ['Shift']) == '@'
    assert components_to_seq('2', ['Control', 'Shift']) == 'Control-Shift-2'

    # test ordering
    assert components_to_seq('2', ['Control', 'Alt', 'Shift', 'Meta']) \
        == 'Control-Alt-Shift-Meta-2'
    assert components_to_seq('2', ['Alt', 'Shift', 'Control', 'Meta']) \
        == 'Control-Alt-Shift-Meta-2'


def test_normalize_key_sequence():
    assert normalize_key_sequence('x') == 'x'
    assert normalize_key_sequence('C-x') == 'Control-x'
    assert normalize_key_sequence('Meta-Alt-x') == 'Alt-Meta-x'
    assert normalize_key_sequence('S-M-C-s-2') == 'Control-Alt-Shift-Meta-2'


def test_determine_binding_op():
    assert _determine_binding_op(None) == BindOperation.UNBIND
    assert _determine_binding_op('SPAM') == BindOperation.REBIND
    assert _determine_binding_op(lambda: 42) == BindOperation.BIND

    with pytest.raises(TypeError):
        _determine_binding_op(1)


def test_unbind_key():
    f = lambda: 42
    kb = dict(a=f)

    assert unbind_key(kb, 'a') is f
    assert kb == {}

    with pytest.raises(KeyError):
        unbind_key(kb, 'a')


def test_rebind_key():
    f = lambda: 42
    kb = dict(a=f)

    rebind_key(kb, 'b', 'a')
    assert kb == dict(b=f)


def test_bind_key():
    kb = {}

    # bind
    f = lambda: 42
    bind_key(kb, 'a', f)
    assert kb == dict(a=f)

    # rebind
    bind_key(kb, 'b', 'a')
    assert kb == dict(b=f)

    # overwrite
    l = lambda: 'SPAM'
    bind_key(kb, 'b', l)
    assert kb == dict(b=l)

    # unbind
    bind_key(kb, 'b', None)
    assert kb == {}


def test_unbind_keys():
    kb = dict(a=lambda: 42,
              b=lambda: 'SPAM',
              c=lambda: 'aliiiens')

    assert [kb['a']] == unbind_keys(kb, ['a'])
    assert kb == dict(b=kb['b'],
                      c=kb['c'])

    assert list(kb.values()) == unbind_keys(kb, list(kb.keys()))
    assert kb == {}


def test_rebind_keys():
    mp = {
        'a': lambda: print(42),
        'b': lambda: print('SPAM!')
    }

    kb = mp.copy()

    # swapped
    rebind_keys(kb, [('a', 'b'),
                     ('b', 'a')])
    assert kb == dict(a=mp['b'], b=mp['a'])

    rebind_keys(kb, [('c', 'b')])
    assert kb == dict(a=mp['b'], c=mp['a'])


def test_bind_keys_normalize_input():
    norm = lambda *args, **kwargs: _bind_keys_normalize_input(args, kwargs)
    out = (('a', 'b'),
           ('c', 'd'))

    assert (norm(('a', 'b'),
                 ('c', 'd'))
            == out)

    assert (norm((('a', 'b'),
                  ('c', 'd')))
            == out)

    assert norm(a='b', c='d') == out

    assert (norm({'a': 'b',
                  'c': 'd'})
            == out)


def test_bind_keys():
    mp = {
        'a': lambda: 42,
        'b': lambda: 'SPAM',
        'c': lambda: 'aliiiens'
    }

    kb = {}

    # bind keys
    bind_keys(kb, mp.items())
    assert kb == mp

    # rebind keys
    bind_keys(kb, [('b', 'a'),
                   ('d', 'b')])
    assert kb['b'] is mp['a']
    assert kb['d'] is mp['b']

    # unbind keys
    bind_keys(kb, [('b', None),
                   ('c', None)])
    assert kb == dict(d=kb['d'])

    # sort/ordering
    bind_keys(kb, [('a', 'd'),
                   ('a', None),
                   ('b', mp['a'])])
    assert kb == dict(a=mp['b'], b=mp['a'])


def test_keybinding_descriptors():
    class Foo:
        default_keybindings = {}

        def __init__(self):
            self.keybindings = self.default_keybindings.copy()

        bind_key = bind_key_method(keybindings='keybindings',
                                   class_keybindings='default_keybindings')
        unbind_key = unbind_key_method(keybindings='keybindings',
                                       class_keybindings='default_keybindings')
        rebind_key = rebind_key_method(keybindings='keybindings',
                                       class_keybindings='default_keybindings')
        bind_keys = bind_keys_method(keybindings='keybindings',
                                     class_keybindings='default_keybindings')

    f = lambda: 42

    Foo.bind_key('a', f)
    assert 'a' in Foo.default_keybindings

    @Foo.bind_key('b')
    def bar(): ...
    assert Foo.default_keybindings['b'] is bar

    assert Foo.unbind_key('b') is bar
    assert list(Foo.default_keybindings.keys()) == ['a']

    Foo.rebind_key('b', 'a')
    assert Foo.default_keybindings == dict(b=f)

    foo = Foo()
    assert foo.keybindings == Foo.default_keybindings

    foo.bind_keys(a=lambda: 'SPAM')
    assert set(foo.keybindings) == {'a', 'b'}
    assert foo.keybindings != Foo.default_keybindings

    f2 = lambda: 'aliiiens'
    foo.bind_keys(('c', f2),
                  ('a', None),
                  ('a', 'b'))
    assert foo.keybindings == dict(a=f,
                                   c=f2)

    foo.bind_keys({'c': None})
    assert foo.keybindings == dict(a=f)

    foo.bind_keys([('a', None)])
    assert foo.keybindings == {}

import os
import sys

from napari.utils.naming import (
    numbered_patt,
    inc_name_count,
    sep,
    start,
    magic_name,
)


def test_re_base_brackets():
    assert numbered_patt.search('layer [12]').group(0) == '12'
    assert numbered_patt.search('layer [e]').group(0) == ''
    assert numbered_patt.search('layer 12]').group(0) == ''
    assert numbered_patt.search('layer [12').group(0) == ''
    assert numbered_patt.search('layer[12]').group(0) == ''
    assert numbered_patt.search('layer 12').group(0) == ''
    assert numbered_patt.search('layer12').group(0) == ''
    assert numbered_patt.search('layer').group(0) == ''


def test_re_other_brackets():
    assert numbered_patt.search('layer [3] [123]').group(0) == '123'


def test_re_first_bracket():
    assert numbered_patt.search(' [42]').group(0) == '42'
    assert numbered_patt.search('[42]').group(0) == '42'


def test_re_sub_base_num():
    assert numbered_patt.sub('8', 'layer [7]', count=1) == 'layer [8]'


def test_re_sub_base_empty():
    assert numbered_patt.sub(' [3]', 'layer', count=1) == 'layer [3]'


def test_inc_name_count():
    assert inc_name_count('layer [7]') == 'layer [8]'
    assert inc_name_count('layer') == f'layer{sep}[{start}]'
    assert inc_name_count('[41]') == '[42]'


os.environ['MAGICNAME'] = '1'
walrus = sys.version_info >= (3, 8)


def test_basic():
    """Check that name is guessed correctly."""

    def inner(x):
        return magic_name(x)

    assert inner(42) is None

    z = 5
    assert inner(z) == 'z'

    if walrus:
        assert eval("inner(y:='SPAM')") == 'y'


globalval = 42


def test_global():
    """Check that it works with global variables."""

    def inner(x):
        return magic_name(x)

    assert inner(globalval) == 'globalval'


def test_function_chains():
    """Check that nothing weird happens with function chains."""

    def inner(x):
        return magic_name(x)

    def foo():
        return 42

    assert inner(foo()) is None


def test_assignment():
    """Check that assignment expressions do not confuse it."""

    def inner(x):
        return magic_name(x)

    result = inner(17)
    assert result is None

    t = 3
    result = inner(t)
    assert inner(t) == 't'

    if walrus:
        result = eval('inner(d:=42)') == 'd'


def test_nesting():
    """Check that nesting works."""

    def outer(x):
        def inner(y):
            return magic_name(y, level=2)

        return inner(x)

    assert outer('literal') is None

    u = 2
    assert outer(u) == 'u'

    if walrus:
        assert eval("outer(e:='aliiiens')") == 'e'


def test_methods():
    """Check that methods work as expected."""

    class Foo:
        def bar(self, z):
            return magic_name(z)

    foo = Foo()

    assert foo.bar('bar') is None

    r = 8
    assert foo.bar(r) == 'r'

    if walrus:
        assert eval('foo.bar(i:=33)') == 'i'

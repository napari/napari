from sys import version_info

from napari.utils.magic import magic_name
from napari.utils import magic

magic.MAGICNAME = 1


walrus = version_info >= (3, 8)


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

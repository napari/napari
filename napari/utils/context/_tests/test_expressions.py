import pytest

from napari.utils.context._expr import Constant, Name


def test_names():
    assert Name("n").eval({'n': 5}) == 5
    with pytest.raises(KeyError):
        assert Name("n").eval({}) is None


def test_constants():
    assert Constant(1).eval() == 1
    assert Constant('asdf').eval() == 'asdf'


def test_bool_ops():
    n1 = Name("n1")
    true = Constant(True)
    false = Constant(False)
    assert (n1 & true).eval({'n1': True}) is True
    assert (n1 & false).eval({'n1': True}) is False
    assert (n1 & false).eval({'n1': False}) is False
    assert (n1 | true).eval({'n1': True}) is True
    assert (n1 | false).eval({'n1': True}) is True
    assert (n1 | false).eval({'n1': False}) is False

    # real constants
    assert (n1 & True).eval({'n1': True}) is True
    assert (n1 & False).eval({'n1': True}) is False
    assert (n1 & False).eval({'n1': False}) is False
    assert (n1 | True).eval({'n1': True}) is True
    assert (n1 | False).eval({'n1': True}) is True
    assert (n1 | False).eval({'n1': False}) is False


def test_comparison():
    n = Name("n")
    n2 = Name("n2")
    one = Constant(1)

    assert (n == n2).eval({'n': 2, 'n2': 2})
    assert not (n == n2).eval({'n': 2, 'n2': 1})
    assert (n != n2).eval({'n': 2, 'n2': 1})
    assert not (n != n2).eval({'n': 2, 'n2': 2})
    # real constant
    assert (n != 1).eval({'n': 2})
    assert not (n != 2).eval({'n': 2})

    assert (n < one).eval({'n': -1})
    assert not (n < one).eval({'n': 2})
    assert (n <= one).eval({'n': 0})
    assert (n <= one).eval({'n': 1})
    assert not (n <= one).eval({'n': 2})
    # with real constant
    assert (n < 1).eval({'n': -1})
    assert not (n < 1).eval({'n': 2})
    assert (n <= 1).eval({'n': 0})
    assert (n <= 1).eval({'n': 1})
    assert not (n <= 1).eval({'n': 2})

    assert (n > one).eval({'n': 2})
    assert not (n > one).eval({'n': 1})
    assert (n >= one).eval({'n': 2})
    assert (n >= one).eval({'n': 1})
    assert not (n >= one).eval({'n': 0})
    # real constant
    assert (n > 1).eval({'n': 2})
    assert not (n > 1).eval({'n': 1})
    assert (n >= 1).eval({'n': 2})
    assert (n >= 1).eval({'n': 1})
    assert not (n >= 1).eval({'n': 0})

from enum import auto
from os.path import abspath, expanduser, sep
from pathlib import Path

import pytest

from napari.utils.misc import (
    StringEnum,
    abspath_or_url,
    callsignature,
    ensure_iterable,
    ensure_sequence_of_iterables,
)

ITERABLE = (0, 1, 2)
NESTED_ITERABLE = [ITERABLE, ITERABLE, ITERABLE]
DICT = {'a': 1, 'b': 3, 'c': 5}
LIST_OF_DICTS = [DICT, DICT, DICT]


@pytest.mark.parametrize(
    'input, expected',
    [
        [ITERABLE, NESTED_ITERABLE],
        [NESTED_ITERABLE, NESTED_ITERABLE],
        [(ITERABLE, (2,), (3, 1, 6)), (ITERABLE, (2,), (3, 1, 6))],
        [DICT, LIST_OF_DICTS],
        [LIST_OF_DICTS, LIST_OF_DICTS],
        [(ITERABLE, (2,), (3, 1, 6)), (ITERABLE, (2,), (3, 1, 6))],
        [None, (None, None, None)],
        # BEWARE: only the first element of a nested sequence is checked.
        [((0, 1), None, None), ((0, 1), None, None)],
    ],
)
def test_sequence_of_iterables(input, expected):
    """Test ensure_sequence_of_iterables returns a sequence of iterables."""
    zipped = zip(range(3), ensure_sequence_of_iterables(input), expected)
    for i, result, expectation in zipped:
        assert result == expectation


def test_sequence_of_iterables_raises():
    with pytest.raises(ValueError):
        # the length argument asserts a specific length
        ensure_sequence_of_iterables(((0, 1),), length=4)

    # BEWARE: only the first element of a nested sequence is checked.
    with pytest.raises(AssertionError):
        iterable = (None, (0, 1), (0, 2))
        result = iter(ensure_sequence_of_iterables(iterable))
        assert next(result) is None


@pytest.mark.parametrize(
    'input, expected',
    [
        [ITERABLE, ITERABLE],
        [DICT, DICT],
        [1, [1, 1, 1]],
        ['foo', ['foo', 'foo', 'foo']],
        [None, [None, None, None]],
    ],
)
def test_ensure_iterable(input, expected):
    """Test test_ensure_iterable returns an iterable."""
    zipped = zip(range(3), ensure_iterable(input), expected)
    for i, result, expectation in zipped:
        assert result == expectation


def test_callsignature():
    # no arguments
    assert str(callsignature(lambda: None)) == '()'

    # one arg
    assert str(callsignature(lambda a: None)) == '(a)'

    # multiple args
    assert str(callsignature(lambda a, b: None)) == '(a, b)'

    # arbitrary args
    assert str(callsignature(lambda *args: None)) == '(*args)'

    # arg + arbitrary args
    assert str(callsignature(lambda a, *az: None)) == '(a, *az)'

    # default arg
    assert str(callsignature(lambda a=42: None)) == '(a=a)'

    # multiple default args
    assert str(callsignature(lambda a=0, b=1: None)) == '(a=a, b=b)'

    # arg + default arg
    assert str(callsignature(lambda a, b=42: None)) == '(a, b=b)'

    # arbitrary kwargs
    assert str(callsignature(lambda **kwargs: None)) == '(**kwargs)'

    # default arg + arbitrary kwargs
    assert str(callsignature(lambda a=42, **kwargs: None)) == '(a=a, **kwargs)'

    # arg + default arg + arbitrary kwargs
    assert str(callsignature(lambda a, b=42, **kw: None)) == '(a, b=b, **kw)'

    # arbitrary args + arbitrary kwargs
    assert str(callsignature(lambda *args, **kw: None)) == '(*args, **kw)'

    # arg + default arg + arbitrary kwargs
    assert (
        str(callsignature(lambda a, b=42, *args, **kwargs: None))
        == '(a, b=b, *args, **kwargs)'
    )

    # kwonly arg
    assert str(callsignature(lambda *, a: None)) == '(a=a)'

    # arg + kwonly arg
    assert str(callsignature(lambda a, *, b: None)) == '(a, b=b)'

    # default arg + kwonly arg
    assert str(callsignature(lambda a=42, *, b: None)) == '(a=a, b=b)'

    # kwonly args + everything
    assert (
        str(callsignature(lambda a, b=42, *, c, d=5, **kwargs: None))
        == '(a, b=b, c=c, d=d, **kwargs)'
    )


def test_string_enum():
    # Make a test StringEnum
    class TestEnum(StringEnum):
        THING = auto()
        OTHERTHING = auto()

    # test setting by value, correct case
    assert TestEnum('thing') == TestEnum.THING

    # test setting by value mixed case
    assert TestEnum('thInG') == TestEnum.THING

    # test setting by instance of self
    assert TestEnum(TestEnum.THING) == TestEnum.THING

    # test setting by name correct case
    assert TestEnum['THING'] == TestEnum.THING

    # test setting by name mixed case
    assert TestEnum['tHiNg'] == TestEnum.THING

    # test setting by value with incorrect value
    with pytest.raises(ValueError):
        TestEnum('NotAThing')

    # test  setting by name with incorrect name
    with pytest.raises(KeyError):
        TestEnum['NotAThing']

    # test creating a StringEnum with the functional API
    animals = StringEnum('Animal', 'AARDVARK BUFFALO CAT DOG')
    assert str(animals.AARDVARK) == 'aardvark'
    assert animals('BUffALO') == animals.BUFFALO
    assert animals['BUffALO'] == animals.BUFFALO

    # test setting by instance of self
    class OtherEnum(StringEnum):
        SOMETHING = auto()

    #  test setting by instance of a different StringEnum is an error
    with pytest.raises(ValueError):
        TestEnum(OtherEnum.SOMETHING)


def test_abspath_or_url():
    relpath = "~" + sep + "something"
    assert abspath_or_url(relpath) == expanduser(relpath)
    assert abspath_or_url('something') == abspath('something')
    assert abspath_or_url(sep + 'something') == abspath(sep + 'something')
    assert abspath_or_url('https://something') == 'https://something'
    assert abspath_or_url('http://something') == 'http://something'
    assert abspath_or_url('ftp://something') == 'ftp://something'
    assert abspath_or_url('s3://something') == 's3://something'
    assert abspath_or_url('file://something') == 'file://something'
    assert abspath_or_url(('a', '~')) == (abspath('a'), expanduser('~'))
    assert abspath_or_url(['a', '~']) == [abspath('a'), expanduser('~')]

    assert abspath_or_url(('a', Path('~'))) == (abspath('a'), expanduser('~'))

    with pytest.raises(TypeError):
        abspath_or_url({'a', '~'})

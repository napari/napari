from enum import auto
from os.path import abspath, expanduser, sep
from pathlib import Path

import pytest

from napari.utils.misc import (
    StringEnum,
    abspath_or_url,
    ensure_iterable,
    ensure_sequence_of_iterables,
    pick_equality_operator,
)

ITERABLE = (0, 1, 2)
NESTED_ITERABLE = [ITERABLE, ITERABLE, ITERABLE]
DICT = {'a': 1, 'b': 3, 'c': 5}
LIST_OF_DICTS = [DICT, DICT, DICT]
PARTLY_NESTED_ITERABLE = [ITERABLE, None, None]
REPEATED_PARTLY_NESTED_ITERABLE = [PARTLY_NESTED_ITERABLE] * 3


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
        [PARTLY_NESTED_ITERABLE, REPEATED_PARTLY_NESTED_ITERABLE],
        [[], ([], [], [])],
    ],
)
def test_sequence_of_iterables(input, expected):
    """Test ensure_sequence_of_iterables returns a sequence of iterables."""
    zipped = zip(
        range(3),
        ensure_sequence_of_iterables(input, repeat_empty=True),
        expected,
    )
    for i, result, expectation in zipped:
        assert result == expectation


def test_sequence_of_iterables_no_repeat_empty():
    assert ensure_sequence_of_iterables([], repeat_empty=False) == []
    with pytest.raises(ValueError):
        ensure_sequence_of_iterables([], repeat_empty=False, length=3)


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

    # test string conversion
    assert str(TestEnum.THING) == 'thing'

    # test direct comparison with a string
    assert TestEnum.THING == 'thing'
    assert 'thing' == TestEnum.THING
    assert TestEnum.THING != 'notathing'
    assert 'notathing' != TestEnum.THING

    # test comparison with another enum with same value names
    class AnotherTestEnum(StringEnum):
        THING = auto()
        ANOTHERTHING = auto()

    assert TestEnum.THING != AnotherTestEnum.THING

    # test lookup in a set
    assert TestEnum.THING in {TestEnum.THING, TestEnum.OTHERTHING}
    assert TestEnum.THING not in {TestEnum.OTHERTHING}
    assert TestEnum.THING in {'thing', TestEnum.OTHERTHING}
    assert TestEnum.THING not in {
        AnotherTestEnum.THING,
        AnotherTestEnum.ANOTHERTHING,
    }


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


def test_equality_operator():
    import operator

    import dask.array as da
    import numpy as np
    import xarray as xr
    import zarr

    class MyNPArray(np.ndarray):
        pass

    assert pick_equality_operator(np.ones((1, 1))) == np.array_equal
    assert pick_equality_operator(MyNPArray([1, 1])) == np.array_equal
    assert pick_equality_operator(da.ones((1, 1))) == operator.is_
    assert pick_equality_operator(zarr.ones((1, 1))) == operator.is_
    assert (
        pick_equality_operator(xr.DataArray(np.ones((1, 1)))) == np.array_equal
    )

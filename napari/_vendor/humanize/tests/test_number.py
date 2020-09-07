#!/usr/bin/env python

"""Number tests."""

import humanize
import pytest
from humanize import number


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("1", "1st"),
        ("2", "2nd"),
        ("3", "3rd"),
        ("4", "4th"),
        ("11", "11th"),
        ("12", "12th"),
        ("13", "13th"),
        ("101", "101st"),
        ("102", "102nd"),
        ("103", "103rd"),
        ("111", "111th"),
        ("something else", "something else"),
        (None, None),
    ],
)
def test_ordinal(test_input, expected):
    assert humanize.ordinal(test_input) == expected


@pytest.mark.parametrize(
    "test_args, expected",
    [
        ([100], "100"),
        ([1000], "1,000"),
        ([10123], "10,123"),
        ([10311], "10,311"),
        ([1_000_000], "1,000,000"),
        ([1_234_567.25], "1,234,567.25"),
        (["100"], "100"),
        (["1000"], "1,000"),
        (["10123"], "10,123"),
        (["10311"], "10,311"),
        (["1000000"], "1,000,000"),
        (["1234567.1234567"], "1,234,567.1234567"),
        ([None], None),
        ([14308.40], "14,308.4"),
        ([14308.40, None], "14,308.4"),
        ([14308.40, 1], "14,308.4"),
        ([14308.40, 2], "14,308.40"),
        ([14308.40, 3], "14,308.400"),
        ([1234.5454545], "1,234.5454545"),
        ([1234.5454545, None], "1,234.5454545"),
        ([1234.5454545, 1], "1,234.5"),
        ([1234.5454545, 2], "1,234.55"),
        ([1234.5454545, 3], "1,234.545"),
        ([1234.5454545, 10], "1,234.5454545000"),
    ],
)
def test_intcomma(test_args, expected):
    assert humanize.intcomma(*test_args) == expected


def test_intword_powers():
    # make sure that powers & human_powers have the same number of items
    assert len(number.powers) == len(number.human_powers)


@pytest.mark.parametrize(
    "test_args, expected",
    [
        (["100"], "100"),
        (["1000000"], "1.0 million"),
        (["1200000"], "1.2 million"),
        (["1290000"], "1.3 million"),
        (["999999999"], "1.0 billion"),
        (["1000000000"], "1.0 billion"),
        (["2000000000"], "2.0 billion"),
        (["999999999999"], "1.0 trillion"),
        (["1000000000000"], "1.0 trillion"),
        (["6000000000000"], "6.0 trillion"),
        (["999999999999999"], "1.0 quadrillion"),
        (["1000000000000000"], "1.0 quadrillion"),
        (["1300000000000000"], "1.3 quadrillion"),
        (["3500000000000000000000"], "3.5 sextillion"),
        (["8100000000000000000000000000000000"], "8.1 decillion"),
        ([None], None),
        (["1230000", "%0.2f"], "1.23 million"),
        ([10 ** 101], "1" + "0" * 101),
    ],
)
def test_intword(test_args, expected):
    assert humanize.intword(*test_args) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (0, "zero"),
        (1, "one"),
        (2, "two"),
        (4, "four"),
        (5, "five"),
        (9, "nine"),
        (10, "10"),
        ("7", "seven"),
        (None, None),
    ],
)
def test_apnumber(test_input, expected):
    assert humanize.apnumber(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (1, "1"),
        (2.0, "2"),
        (4.0 / 3.0, "1 1/3"),
        (5.0 / 6.0, "5/6"),
        ("7", "7"),
        ("8.9", "8 9/10"),
        ("ten", "ten"),
        (None, None),
        (1 / 3, "1/3"),
        (1.5, "1 1/2"),
        (0.3, "3/10"),
        (0.333, "333/1000"),
    ],
)
def test_fractional(test_input, expected):
    assert humanize.fractional(test_input) == expected


@pytest.mark.parametrize(
    "test_args, expected",
    [
        ([1000], "1.00 x 10³"),
        ([-1000], "1.00 x 10⁻³"),
        ([5.5], "5.50 x 10⁰"),
        ([5781651000], "5.78 x 10⁹"),
        (["1000"], "1.00 x 10³"),
        (["99"], "9.90 x 10¹"),
        ([float(0.3)], "3.00 x 10⁻¹"),
        (["foo"], "foo"),
        ([None], None),
        ([1000, 1], "1.0 x 10³"),
        ([float(0.3), 1], "3.0 x 10⁻¹"),
        ([1000, 0], "1 x 10³"),
        ([float(0.3), 0], "3 x 10⁻¹"),
    ],
)
def test_scientific(test_args, expected):
    assert humanize.scientific(*test_args) == expected

#!/usr/bin/env python

"""Tests for filesize humanizing."""

import humanize
import pytest


@pytest.mark.parametrize(
    "test_args, expected",
    [
        ([300], "300 Bytes"),
        ([3000], "3.0 kB"),
        ([3000000], "3.0 MB"),
        ([3000000000], "3.0 GB"),
        ([3000000000000], "3.0 TB"),
        ([300, True], "300 Bytes"),
        ([3000, True], "2.9 KiB"),
        ([3000000, True], "2.9 MiB"),
        ([300, False, True], "300B"),
        ([3000, False, True], "2.9K"),
        ([3000000, False, True], "2.9M"),
        ([1024, False, True], "1.0K"),
        ([10 ** 26 * 30, False, True], "2481.5Y"),
        ([10 ** 26 * 30, True], "2481.5 YiB"),
        ([10 ** 26 * 30], "3000.0 YB"),
        ([1, False, False], "1 Byte"),
        ([3141592, False, False, "%.2f"], "3.14 MB"),
        ([3000, False, True, "%.3f"], "2.930K"),
        ([3000000000, False, True, "%.0f"], "3G"),
        ([10 ** 26 * 30, True, False, "%.3f"], "2481.542 YiB"),
    ],
)
def test_naturalsize(test_args, expected):
    assert humanize.naturalsize(*test_args) == expected

    args_with_negative = test_args
    args_with_negative[0] *= -1
    assert humanize.naturalsize(*args_with_negative) == "-" + expected

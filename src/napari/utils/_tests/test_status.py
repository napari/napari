import numpy as np
import pytest

from napari.settings import get_settings
from napari.utils.status_messages import status_format

STRING = 'hello world'
STRING_FORMATTED = STRING
MISSING = None
MISSING_FORMATTED = ''
NUMERIC = [
    1,
    10,
    100,
    1000,
    1e6,
    -6.283,
    123.932021,
    1123.9392001,
    2 * np.pi,
    np.exp(1),
]
NUMERIC_FORMATTED = (
    '[1, 10, 100, 1000, 1e+06, -6.28, 124, 1.12e+03, 6.28, 2.72]'
)
COMBINED = [1e6, MISSING, STRING]
COMBINED_FORMATTED = f'[1e+06, {MISSING_FORMATTED}, {STRING_FORMATTED}]'


@pytest.mark.parametrize(
    ('input_data', 'expected'),
    [
        (NUMERIC, NUMERIC_FORMATTED),
        (STRING, STRING_FORMATTED),
        (MISSING, MISSING_FORMATTED),
        (COMBINED, COMBINED_FORMATTED),
    ],
)
def test_status_format(input_data, expected, monkeypatch):
    """test various formatting cases embodied in utils.status_messages.status_format with the default parameter 3"""

    monkeypatch.setattr(
        get_settings().application, 'float_display_precision', 3
    )
    assert status_format(input_data) == expected


def test_status_format_precision_setting(monkeypatch):
    """test that the float_display_precision setting is respected"""
    monkeypatch.setattr(
        get_settings().application, 'float_display_precision', 2
    )
    assert (
        status_format(
            123.932021,
        )
        == '1.2e+02'
    )

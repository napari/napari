import numpy as np
import pytest

from napari.utils.status_messages import status_format

STRING = "hello world"
STRING_FORMATTED = STRING
MISSING = None
MISSING_FORMATTED = ""
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
    "[1, 10, 100, 1000, 1e+06, -6.28, 124, 1.12e+03, 6.28, 2.72]"
)
COMBINED = [1e6, MISSING, STRING]
COMBINED_FORMATTED = f"[1e+06, {MISSING_FORMATTED}, {STRING_FORMATTED}]"


@pytest.mark.parametrize(
    'input_data, expected',
    [
        [NUMERIC, NUMERIC_FORMATTED],
        [STRING, STRING_FORMATTED],
        [MISSING, MISSING_FORMATTED],
        [COMBINED, COMBINED_FORMATTED],
    ],
)
def test_status_format(input_data, expected):
    """test various formatting cases embodied in utils.status_messages.status_format"""

    assert status_format(input_data) == expected

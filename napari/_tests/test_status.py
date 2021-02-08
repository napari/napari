import numpy as np
import pytest

from napari.utils.status_messages import status_format


def test_status_format(make_napari_viewer):
    """ test various formatting cases embodied in utils.status_messages.status_format """

    values = np.array(
        [
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
    )
    assert (
        status_format(values)
        == '[1, 10, 100, 1e+03, 1e+06, -6.28, 124, 1.12e+03, 6.28, 2.72]'
    )
    assert status_format('hello') == 'hello'
    assert (
        status_format([1.23, -1.23, None, 4.5789]) == '[1.23, -1.23, , 4.58]'
    )

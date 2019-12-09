import numpy as np
import pytest
from dask import array as da

from ..layer_utils import calc_data_range


data_dask = da.random.random(
    size=(100_000, 1000, 1000), chunks=(1, 1000, 1000)
)


def test_calc_data_range():
    # all zeros should return [0, 1] by default
    data = np.zeros((10, 10))
    clim = calc_data_range(data)
    assert np.all(clim == [0, 1])

    # all ones should return [0, 1] by default
    data = np.ones((10, 10))
    clim = calc_data_range(data)
    assert np.all(clim == [0, 1])

    # return min and max
    data = np.random.random((10, 15))
    data[0, 0] = 0
    data[0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])

    # return min and max
    data = np.random.random((6, 10, 15))
    data[0, 0, 0] = 0
    data[0, 0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])

    # Try large data
    data = np.zeros((1000, 2000))
    data[0, 0] = 0
    data[0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])

    # Try large data mutlidimensional
    data = np.zeros((3, 1000, 1000))
    data[0, 0, 0] = 0
    data[0, 0, 1] = 2
    clim = calc_data_range(data)
    assert np.all(clim == [0, 2])


def test_calc_data_fast_uint8():
    data = da.random.randint(
        0,
        100,
        size=(100_000, 1000, 1000),
        chunks=(1, 1000, 1000),
        dtype=np.uint8,
    )
    assert calc_data_range(data) == [0, 255]


@pytest.mark.timeout(2)
def test_calc_data_range_fast_big():
    val = calc_data_range(data_dask)
    assert len(val) > 0

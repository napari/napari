import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.util import img_as_ubyte

from napari.layers.utils.layer_utils import convert_to_uint8


@pytest.mark.filterwarnings("ignore:Downcasting uint:UserWarning:skimage")
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_uint(dtype):
    data = np.arange(50, dtype=dtype)
    data_scaled = data * 256 ** (data.dtype.itemsize - 1)
    assert convert_to_uint8(data_scaled).dtype == np.uint8
    assert_array_equal(data, convert_to_uint8(data_scaled))
    assert_array_equal(img_as_ubyte(data), convert_to_uint8(data))
    assert_array_equal(
        img_as_ubyte(data_scaled), convert_to_uint8(data_scaled)
    )


@pytest.mark.filterwarnings("ignore:Downcasting int:UserWarning:skimage")
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_int(dtype):
    data = np.arange(50, dtype=dtype)
    data_scaled = data * 256 ** (data.dtype.itemsize - 1)
    assert convert_to_uint8(data).dtype == np.uint8
    assert convert_to_uint8(data_scaled).dtype == np.uint8
    assert_array_equal(img_as_ubyte(data), convert_to_uint8(data))
    assert_array_equal(2 * data, convert_to_uint8(data_scaled))
    assert_array_equal(
        img_as_ubyte(data_scaled), convert_to_uint8(data_scaled)
    )
    assert_array_equal(img_as_ubyte(data - 10), convert_to_uint8(data - 10))
    assert_array_equal(
        img_as_ubyte(data_scaled - 10), convert_to_uint8(data_scaled - 10)
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32, float])
def test_float(dtype):
    data = np.linspace(0, 0.5, 128, dtype=dtype, endpoint=False)
    res = np.arange(128, dtype=np.uint8)
    assert convert_to_uint8(data).dtype == np.uint8
    assert_array_equal(convert_to_uint8(data), res)
    data = np.linspace(0, 1, 256, dtype=dtype)
    res = np.arange(256, dtype=np.uint8)
    assert_array_equal(convert_to_uint8(data), res)
    assert_array_equal(img_as_ubyte(data), convert_to_uint8(data))
    assert_array_equal(img_as_ubyte(data - 0.5), convert_to_uint8(data - 0.5))


def test_bool():
    data = np.zeros((10, 10), dtype=bool)
    data[2:-2, 2:-2] = 1
    converted = convert_to_uint8(data)
    assert converted.dtype == np.uint8
    assert_array_equal(img_as_ubyte(data), converted)

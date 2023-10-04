import numpy as np
import pytest
from numpy.testing import (
    assert_array_equal,
    assert_raises,
)

from napari.experimental import _progressive_loading, _virtual_data
from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset,
)


@pytest.fixture
def max_level():
    return 8


@pytest.fixture
def mandelbrot_arrays(max_level):
    large_image = mandelbrot_dataset(max_levels=max_level)
    multiscale_img = large_image["arrays"]
    return multiscale_img


def test_virtualdata_init(mandelbrot_arrays, max_level):
    scale = max_level - 1
    _virtual_data.VirtualData(mandelbrot_arrays[scale], scale=scale)


def test_virtualdata_set_interval(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _virtual_data.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(513, 1024, None), slice(513, 1024, None))
    vdata.set_interval(coords)

    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]

    # vdata slices are based on chunks so we make sure that the requested
    # coordinates fall within the chunked slice
    assert vdata._min_coord <= min_coord
    assert vdata._max_coord >= max_coord


def test_virtualdata_getitem(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _virtual_data.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    retrieved_data = vdata[(500, 500)]
    # Ensure the retrieved data matches expected value
    print(vdata[0:1024, 0:1024])
    assert retrieved_data == 0


def test_virtualdata_hyperslice_reuse(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _virtual_data.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    first_hyperslice = vdata.hyperslice
    vdata.set_interval(coords)
    second_hyperslice = vdata.hyperslice
    assert_array_equal(first_hyperslice, second_hyperslice)


def test_virtualdata_hyperslice(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _virtual_data.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    first_hyperslice = vdata.hyperslice
    coords = (slice(513, 1024, None), slice(513, 1024, None))
    # the width in set_interval is calculated to be 1024, therefore the first and second hyperslice are equal, causing this test to fail
    vdata.set_interval(coords)
    second_hyperslice = vdata.hyperslice
    assert_raises(
        AssertionError, assert_array_equal, first_hyperslice, second_hyperslice
    )


def test_get_offset_default(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _virtual_data.VirtualData(mandelbrot_arrays[scale], scale=scale)

    # Interval not set - should return zeros
    assert_array_equal(vdata[100, 100], np.zeros((1, 1)))


def test_multiscalevirtualdata_init(mandelbrot_arrays):
    mvdata = _virtual_data.MultiScaleVirtualData(mandelbrot_arrays)
    assert isinstance(mvdata, _progressive_loading.MultiScaleVirtualData)


def test_multiscalevirtualdata_set_interval(mandelbrot_arrays):
    mvdata = _virtual_data.MultiScaleVirtualData(mandelbrot_arrays)
    coords = (slice(513, 1024, None), slice(513, 1024, None))
    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]
    mvdata.set_interval(min_coord=min_coord, max_coord=max_coord)

    # mvdata slices are based on chunks so we make sure that the requested
    # coordinates fall within the chunked slice
    assert mvdata._data[0]._min_coord <= min_coord
    assert mvdata._data[0]._max_coord <= max_coord


def test_multiscale_set_interval(mandelbrot_arrays):
    mvdata = _virtual_data.MultiScaleVirtualData(mandelbrot_arrays)
    mvdata.set_interval([0, 0], [1024, 1024])

    assert mvdata._data[0]._min_coord == [0, 0]
    assert mvdata._data[1]._min_coord == [0, 0]

    assert mvdata._data[0]._max_coord == [1024, 1024]
    assert mvdata._data[1]._max_coord == [512, 512]


# def test_nd_array():

#     shape = (10, 20, 30, 40)
#     chunks = (1, 10, 15, 20)
#     arr = da.random.random(shape, chunks=chunks)

#     vdata = _virtual_data.VirtualData(arr, 0, ndisplay=3)

#     coords = tuple(slice(None) for _ in shape)
#     vdata.set_interval(coords)

#     assert vdata.hyperslice.shape == shape[-2:]
#     assert isinstance(vdata.hyperslice, da.Array)


# def test_multiscale_nd_array():

#     shape = (10, 20, 30, 40)
#     chunks = (5, 10, 15, 20)
#     num_scales = 5

#     arr_list = [da.random.rand(shape, chunks=chunks) for _ in range(num_scales)]

#     mvdata = _virtual_data.MultiScaleVirtualData(arr_list, ndisplay=2)

#     min_coord = [0]*len(shape)
#     max_coord = [s - 1 for s in shape]

#     mvdata.set_interval(min_coord, max_coord)

#     for vdata in mvdata._data:
#         assert vdata.hyperslice.shape == shape[-2:]
#         assert isinstance(vdata.hyperslice, da.Array)

if __name__ == "__main__":
    max_level = 8

    large_image = mandelbrot_dataset(max_levels=8)
    multiscale_img = large_image["arrays"]

    scale = max_level - 1
    vdata = _virtual_data.VirtualData(multiscale_img[-1], scale=scale)
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    retrieved_data = np.asarray(vdata[0:1024, 0:1024])
    # Ensure the retrieved data matches expected value
    print(retrieved_data)

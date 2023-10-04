import dask.array as da
import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)

import napari
from napari.experimental import _progressive_loading
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
    chunk_centers,
    chunk_slices,
    distance_from_camera_center_line,
    get_chunk,
    visual_depth,
)
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
    _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)


def test_virtualdata_set_interval(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    coords = (slice(513, 1024, None), slice(513, 1024, None))
    vdata.set_interval(coords)

    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]

    # vdata slices are based on chunks so we make sure that the requested
    # coordinates fall within the chunked slice
    assert vdata._min_coord <= min_coord
    assert vdata._max_coord >= max_coord


def test_virtualdata_hyperslice_reuse(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    first_hyperslice = vdata.hyperslice
    vdata.set_interval(coords)
    second_hyperslice = vdata.hyperslice
    assert_array_equal(first_hyperslice, second_hyperslice)


def test_virtualdata_hyperslice(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
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


def test_multiscalevirtualdata_init(mandelbrot_arrays):
    mvdata = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)
    assert isinstance(mvdata, _progressive_loading.MultiScaleVirtualData)


def test_multiscalevirtualdata_set_interval(mandelbrot_arrays):
    mvdata = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)
    coords = (slice(513, 1024, None), slice(513, 1024, None))
    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]
    mvdata.set_interval(min_coord=min_coord, max_coord=max_coord)

    # mvdata slices are based on chunks so we make sure that the requested
    # coordinates fall within the chunked slice
    assert mvdata._data[0]._min_coord <= min_coord
    assert mvdata._data[0]._max_coord <= max_coord


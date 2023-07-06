import numpy as np
import pytest
from _mandelbrot_vizarr import add_progressive_loading_image
from numpy.testing import assert_array_equal, assert_raises

import napari
from napari.experimental import _progressive_loading
from napari.experimental._progressive_loading import get_chunk
from napari.experimental._progressive_loading_datasets import (
    MandlebrotStore,
    mandelbrot_dataset,
)


@pytest.fixture
def max_level():
    return 14

@pytest.fixture
def mandelbrot_arrays(max_level):
    large_image = mandelbrot_dataset(max_levels=max_level)
    multiscale_img = large_image["arrays"]
    return multiscale_img

def test_add_progressive_loading_image(mandelbrot_arrays):
    viewer = napari.Viewer()
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)


def test_add_progressive_loading_image_zoom_in(mandelbrot_arrays):
    viewer = napari.Viewer()
    viewer.camera.zoom = 0.0001
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = 0.001  # only fails if we change visible scales


def test_add_progressive_loading_image_zoom_out(mandelbrot_arrays):
    viewer = napari.Viewer()
    viewer.camera.zoom = 0.001
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = 0.0001  # only fails if we change visible scales


def test_chunk_slices_0_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    data_interval = np.array([[0, 0], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(0, 512, None), slice(512, 1024, None)],
        [slice(0, 512, None), slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result

def test_chunk_slices_512_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    data_interval = np.array([[512, 512], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)],
        [slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result

def test_chunk_slices_600_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    data_interval = np.array([[600, 512], [600, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)],
        [slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result


def test_virtualdata_init(mandelbrot_arrays, max_level):
    scale = max_level - 1
    _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)


def test_virtualdata_set_interval(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(512, 1024, None), slice(512, 1024, None))
    vdata.set_interval(coords)

    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]
    assert vdata._min_coord == min_coord
    assert vdata._max_coord == max_coord

def test_virtualdata_hyperslice_reuse(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    first_hyperslice = vdata.hyperslice
    vdata.set_interval(coords)
    second_hyperslice = vdata.hyperslice
    assert_array_equal(first_hyperslice, second_hyperslice)


def test_virtualdata_hyperslice(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    first_hyperslice = vdata.hyperslice
    coords = (slice(512, 1024, None), slice(512, 1024, None))
    vdata.set_interval(coords)
    second_hyperslice = vdata.hyperslice
    assert_raises(AssertionError, assert_array_equal, first_hyperslice, second_hyperslice)


def test_multiscalevirtualdata_init(mandelbrot_arrays):
    mvdata = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)
    assert isinstance(mvdata, _progressive_loading.MultiScaleVirtualData)


@pytest.mark.parametrize('max_level', [8, 14])
def test_MandlebrotStore(max_level):
    MandlebrotStore(
        levels=max_level, tilesize=512, compressor=None, maxiter=255
    )

def test_get_chunk(mandelbrot_arrays):
    scale = 12
    virtual_data = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    chunk_slice = (slice(1024, 1536, None), slice(512, 1024, None))

    chunk_widths = (chunk_slice[0].stop - chunk_slice[0].start, chunk_slice[1].stop - chunk_slice[1].start)
    real_array = get_chunk(chunk_slice, array=virtual_data)

    assert chunk_widths == real_array.shape


if __name__ == "__main__":
    viewer = napari.Viewer()
    large_image = mandelbrot_dataset(max_levels=14)
    mandelbrot_arrays = large_image["arrays"]

    scale = 7
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)

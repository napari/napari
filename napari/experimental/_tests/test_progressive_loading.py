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


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_add_progressive_loading_image(mandelbrot_arrays):
    viewer = napari.Viewer()
    # pytest.warns()
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_add_progressive_loading_image_zoom_in(mandelbrot_arrays):
    viewer = napari.Viewer()
    viewer.camera.zoom = 0.0001
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = 0.001  # only fails if we change visible scales


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_add_progressive_loading_image_zoom_out(mandelbrot_arrays):
    viewer = napari.Viewer()
    viewer.camera.zoom = 0.001
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = 0.0001  # only fails if we change visible scales


def test_chunk_slices_0_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    data_interval = np.array([[0, 0], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(
        vdata, interval=data_interval
    )
    dims = len(vdata.array.shape)

    result = [
        [slice(0, 512, None), slice(512, 1024, None)],
        [slice(0, 512, None), slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result


def test_chunk_slices_512_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    data_interval = np.array([[512, 512], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(
        vdata, interval=data_interval
    )
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)],
        [slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result


def test_chunk_slices_600_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    data_interval = np.array([[600, 512], [600, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(
        vdata, interval=data_interval
    )
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


@pytest.mark.xfail(
    reason='shouldnt fail, the width in set_interval is calculated to be 1024, therefore the first and second hyperslice are equal'
)
def test_virtualdata_hyperslice_fail(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    coords = (slice(0, 1024, None), slice(0, 1024, None))
    vdata.set_interval(coords)
    first_hyperslice = vdata.hyperslice
    coords = (slice(512, 1024, None), slice(512, 1024, None))
    # the width in set_interval is calculated to be 1024, therefore the first and second hyperslice are equal, causing this test to fail
    vdata.set_interval(coords)
    second_hyperslice = vdata.hyperslice
    assert_raises(
        AssertionError, assert_array_equal, first_hyperslice, second_hyperslice
    )


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


def test_get_chunk(mandelbrot_arrays, max_level):
    scale = max_level - 1
    virtual_data = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    chunk_slice = (slice(1024, 1536, None), slice(512, 1024, None))

    chunk_widths = (
        chunk_slice[0].stop - chunk_slice[0].start,
        chunk_slice[1].stop - chunk_slice[1].start,
    )
    real_array = get_chunk(chunk_slice, array=virtual_data)

    assert chunk_widths == real_array.shape


def test_visual_depth(make_napari_viewer):
    """TODO: this test passes if you run it by itself"""
    viewer = make_napari_viewer()
    points = np.array([10, 50, 20])

    viewer.camera.set_view_direction((1, 0, 0))
    projected_length = visual_depth(points, viewer.camera)
    assert_array_almost_equal(projected_length, points[0])

    viewer.camera.set_view_direction((0, 1, 0))
    projected_length = visual_depth(points, viewer.camera)
    assert_array_almost_equal(projected_length, points[1])

    with pytest.warns(UserWarning, match='Gimbal lock detected'):
        viewer.camera.set_view_direction((0, 0, 1))
        projected_length = visual_depth(points, viewer.camera)
        assert_array_almost_equal(projected_length, points[2])
    # qtbot.wait(5000) # wait for asyncronous tasks to complete, doesn't solve the problem


def test_distance_from_camera_center_line(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.array([10, 0, 0])

    viewer.camera.set_view_direction((1, 0, 0))
    distances = distance_from_camera_center_line(points, viewer.camera)
    assert_array_almost_equal(distances, np.array([0]))

    viewer.camera.set_view_direction((0, 1, 0))
    distances = distance_from_camera_center_line(points, viewer.camera)
    assert_array_almost_equal(distances, np.array([10]))


def test_chunk_centers_2d():
    dask_arr = da.random.random((20, 20), chunks=(10, 10))
    mapping = chunk_centers(dask_arr, ndim=2)
    centers = list(mapping.keys())
    expected = [(5.0, 5.0), (5.0, 15.0), (15.0, 5.0), (15.0, 15.0)]

    assert centers == expected


def test_chunk_centers_3d():
    dask_arr = da.random.random((20, 20, 20), chunks=(10, 20, 20))
    mapping = chunk_centers(dask_arr, ndim=3)
    centers = list(mapping.keys())
    expected = [(5.0, 10.0, 10.0), (15.0, 10.0, 10.0)]

    assert centers == expected


def test_chunk_slices_no_interval():
    dask_arr = da.random.random((20, 20, 20), chunks=(10, 20, 20))
    slices = chunk_slices(dask_arr, interval=None)
    expected = [
        [slice(0, 10, None), slice(10, 20, None)],
        [slice(0, 20, None)],
        [slice(0, 20, None)],
    ]

    assert slices == expected


def test_chunk_slices_with_interval():
    dask_arr = da.random.random((20, 20, 20), chunks=(10, 20, 20))
    interval = np.array([[0, 0, 0], [20, 20, 20]])
    slices = chunk_slices(dask_arr, interval=interval)
    expected = [
        [slice(0, 10, None), slice(10, 20, None)],
        [slice(0, 20, None)],
        [slice(0, 20, None)],
    ]

    assert slices == expected

    interval = np.array([[11, 11, 11], [20, 20, 20]])
    slices = chunk_slices(dask_arr, interval=interval)
    expected = [
        [slice(10, 20, None)],
        [slice(0, 20, None)],
        [slice(0, 20, None)],
    ]

    assert slices == expected

    interval = np.array([[10, 10, 10], [20, 20, 20]])
    slices = chunk_slices(dask_arr, interval=interval)
    expected = [
        [slice(0, 10, None), slice(10, 20, None)],
        [slice(0, 20, None)],
        [slice(0, 20, None)],
    ]

    assert slices == expected


if __name__ == "__main__":
    viewer = napari.Viewer()
    max_level = 8

    large_image = mandelbrot_dataset(max_levels=max_level)
    mandelbrot_arrays = large_image["arrays"]

    scale = 7
    vdata = _progressive_loading.VirtualData(
        mandelbrot_arrays[scale], scale=scale
    )
    mvdata = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)

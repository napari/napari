import logging
import sys
import pytest 
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

import napari

from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset, MandlebrotStore
)
from napari.experimental import _progressive_loading

from _mandelbrot_vizarr import add_progressive_loading_image

# config.async_loading = True

LOGGER = logging.getLogger("mandelbrot_vizarr")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


@pytest.fixture
def mandelbrot_image():
    large_image = mandelbrot_dataset()
    multiscale_img = large_image["arrays"]
    return multiscale_img

@pytest.fixture
def mandelbrot_arrays():
    large_image = mandelbrot_dataset()
    multiscale_img = large_image["arrays"]
    return multiscale_img

# def test_get_chunk():
#     pass

# def test_visual_depth():
#     pass

# def test_distance_from_camera_centre_line():
#     pass

# def test_chunk_centers():
#     pass

def test_chunk_slices_0_1024(mandelbrot_arrays):
    scale = 7
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    data_interval = np.array([[0, 0], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(0, 512, None), slice(512, 1024, None)],
        [slice(0, 512, None), slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result

def test_chunk_slices_512_1024(mandelbrot_arrays):
    scale = 7
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    data_interval = np.array([[512, 512], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)],
        [slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result

def test_chunk_slices_600_1024(mandelbrot_arrays):
    scale = 7
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    data_interval = np.array([[600, 512], [600, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)], 
        [slice(512, 1024, None)], 
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result


# def test_chunk_priority_2D():
#     pass

# def test_prioritised_chunk_loading_3D():
#     pass

# def test_render_sequence_3D_caller():
#     pass

# def test_render_sequence_3D():
#     pass

# def test_interpolated_get_chunk_2D():
#     pass



def test_virtualdata_init(mandelbrot_arrays):
    scale = 0
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    

def test_virtualdata_set_interval(mandelbrot_arrays):
    scale = 0
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    coords = tuple([slice(512, 1024, None), slice(512, 1024, None)])
    vdata.set_interval(coords)

    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]
    assert vdata._min_coord == min_coord
    assert vdata._max_coord == max_coord

def test_virtualdata_data_plane_reuse(mandelbrot_arrays):
    scale = 0
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    coords = tuple([slice(0, 1024, None), slice(0, 1024, None)])
    vdata.set_interval(coords)
    first_data_plane = vdata.data_plane
    vdata.set_interval(coords)
    second_data_plane = vdata.data_plane
    assert_array_equal(first_data_plane, second_data_plane)


def test_virtualdata_data_plane(mandelbrot_arrays):
    scale = 0
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    coords = tuple([slice(0, 1024, None), slice(0, 1024, None)])
    vdata.set_interval(coords)
    first_data_plane = vdata.data_plane
    coords = tuple([slice(512, 1024, None), slice(512, 1024, None)])
    vdata.set_interval(coords)
    second_data_plane = vdata.data_plane
    assert_raises(AssertionError, assert_array_equal, first_data_plane, second_data_plane)


def test_multiscalevirtualdata_init(mandelbrot_arrays):
    mvdata = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)
    assert isinstance(mvdata, _progressive_loading.MultiScaleVirtualData)


def test_MandlebrotStore():
    max_levels=8
    store = MandlebrotStore(
        levels=max_levels, tilesize=512, compressor=None, maxiter=255  
    ) 

# def test_cause_crash():
#     """This test causes this error:
#     ```
#     RuntimeError: Workers did not quit gracefully in the time allotted (5000 ms)
#     ```
#     """
#     viewer = napari.Viewer()

#     # large_image = openorganelle_mouse_kidney_em()
#     large_image = mandelbrot_dataset()

#     multiscale_img = large_image["arrays"]
#     viewer._layer_slicer._force_sync = False

#     add_progressive_loading_image(multiscale_img, viewer=viewer)

#     viewer.camera.zoom = 0.01
#     napari.run()


# def test_data_plane():
#     viewer = napari.Viewer()

#     # large_image = openorganelle_mouse_kidney_em()
#     large_image = mandelbrot_dataset()

#     multiscale_img = large_image["arrays"]
#     viewer._layer_slicer._force_sync = False

#     add_progressive_loading_image(multiscale_img, viewer=viewer)


#     napari.run()


    

if __name__ == "__main__":
    viewer = napari.Viewer()
    large_image = mandelbrot_dataset()
    mandelbrot_arrays = large_image["arrays"]
    # scale = 7
    # vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    # # canvas corners
    # top_left = [0, 0]
    # bottom_right = [365535, 421535]
    # scaled_min = [0, 0]
    # scaled_max = [1024, 1024]
    # # coords = tuple([slice(0, 0), slice(512, 512)])
    # coords = tuple([slice(0, 1024, None), slice(0, 1024, None)])
    # vdata.set_interval(coords)
    # first_data_plane = vdata.data_plane
    # vdata.set_interval(coords)
    # second_data_plane = vdata.data_plane

    # assert_array_equal(first_data_plane, second_data_plane)


    scale = 7
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale])
    # corner_pixels # canvas coordinates
    # data_interval = corner_pixels / (2**scale)  # data interval
    data_interval = np.array([[512, 512], [1024, 1024]])
    # data_interval = np.array([[0, 0], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)
    # result = [
    #     slice(512, 1024, None), slice(512, 1024, None),
    #     slice(512, 1024, None), slice(512, 1024, None),
    # ]
    result = [
        [slice(0, 512, None), slice(512, 1024, None)],
        [slice(0, 512, None), slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result
    # viewer = napari.Viewer()

    # # large_image = openorganelle_mouse_kidney_em()
    # large_image = mandelbrot_dataset()

    # multiscale_img = large_image["arrays"]
    # viewer._layer_slicer._force_sync = False

    # add_progressive_loading_image(multiscale_img, viewer=viewer)


    # napari.run()


#     2023-05-22 11:02:08,101 - napari.experimental._progressive_loading - INFO - 
# MultiscaleVirtualData: update_with_minmax: scale 7 min [0 0] : [0, 0] max [131072 131072] : 
# scaled max [1024, 1024]
# 2023-05-22 11:02:08,101 - napari.experimental._progressive_loading - DEBUG - 
# VirtualData: update_with_minmax: [0, 0] max [1024, 1024] interval size [1024, 1024]

# 0.007113208142185655
# viewer.camera.center = (0.0, 44313.950610539905, 69661.23319329295)
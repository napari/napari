from typing import List, Union

import numpy as np
import pytest
from vispy.visuals import VolumeVisual

from napari import Viewer
from napari._vispy.layers.image import VispyImageLayer
from napari.utils.events import Event

# The tests in this module for the new style of async slicing in napari:
# https://napari.org/dev/naps/4-async-slicing.html
# They are marked with sync_only because that denotes that the old experimental
# async should not run as we don't explicitly wait for its threads to finish.


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.sync_only
def test_async_slice_image_on_current_step_change(
    make_napari_viewer, qtbot, rng
):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    vispy_image = setup_viewer_for_async_slice_image(viewer, data)
    assert viewer.dims.current_step != (2, 0, 0)

    viewer.dims.current_step = (2, 0, 0)

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[2, :, :])


@pytest.mark.sync_only
def test_async_slice_image_on_order_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    vispy_image = setup_viewer_for_async_slice_image(viewer, data)
    assert viewer.dims.order != (1, 0, 2)

    viewer.dims.order = (1, 0, 2)

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[:, 2, :])


@pytest.mark.sync_only
def test_async_slice_image_on_ndisplay_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    vispy_image = setup_viewer_for_async_slice_image(viewer, data)
    assert viewer.dims.ndisplay != 3

    viewer.dims.ndisplay = 3

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data)


@pytest.mark.sync_only
def test_async_slice_multiscale_image_on_pan(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    vispy_image = setup_viewer_for_async_slice_image(viewer, data)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 2
    image = vispy_image.layer
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 4, 5]])

    # Simulate panning to the left by changing the corner pixels in the last
    # dimension, which corresponds to x/columns, then triggering a reload.
    image.corner_pixels = np.array([[0, 0, 0], [0, 4, 3]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[1][1, 0:4, 0:3])


@pytest.mark.sync_only
def test_async_slice_multiscale_image_on_zoom(qtbot, make_napari_viewer, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    vispy_image = setup_viewer_for_async_slice_image(viewer, data)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 2
    image = vispy_image.layer
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 4, 5]])

    # Simulate zooming into the middle of the higher resolution image.
    image._data_level = 0
    image.corner_pixels = np.array([[0, 2, 3], [0, 6, 7]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[0][2, 2:6, 3:7])


def setup_viewer_for_async_slice_image(
    viewer: Viewer,
    data: Union[np.ndarray, List[np.ndarray]],
) -> VispyImageLayer:
    # Initially force synchronous slicing so any slicing caused
    # by adding the image finishes before any other slicing starts.
    viewer._layer_slicer._force_sync = True
    # Add the image and get the corresponding vispy image.
    layer = viewer.add_image(data)
    vispy_layer = viewer.window._qt_viewer.layer_to_visual[layer]
    # Then allow asynchronous slicing for testing.
    viewer._layer_slicer._force_sync = False
    return vispy_layer


def wait_until_vispy_image_data_equal(
    qtbot, vispy_layer: VispyImageLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_image_data_equal() -> None:
        node = vispy_layer.node
        data = (
            node._last_data if isinstance(node, VolumeVisual) else node._data
        )
        # Vispy node data may have been post-processed (e.g. through a colormap),
        # so check that values are close rather than exactly equal.
        np.testing.assert_allclose(data, expected_data)

    qtbot.waitUntil(assert_vispy_image_data_equal)

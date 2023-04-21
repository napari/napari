# The tests in this module for the new style of async slicing in napari:
# https://napari.org/dev/naps/4-async-slicing.html
from threading import RLock

import numpy as np
import pytest
from vispy.visuals import VolumeVisual

from napari import Viewer
from napari._vispy.layers.image import VispyImageLayer
from napari._vispy.visuals.points import PointsVisual
from napari.layers import Image, Layer, Points
from napari.utils.events import Event


class LockableImage(Image):
    """Lockable version of Image. This allows us to assert state and
    conditions that may only be temporarily true at different stages of
    an asynchronous task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock: RLock = RLock()


class LockablePoints(Points):
    """Lockable version of Points. This allows us to assert state and
    conditions that may only be temporarily true at different stages of
    an asynchronous task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock: RLock = RLock()


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_async_slice_image_on_current_step_change(
    make_napari_viewer, qtbot, rng
):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    layer = Image(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)

    assert viewer.dims.current_step != (2, 0, 0)

    viewer.dims.current_step = (2, 0, 0)

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data[2, :, :])


def test_async_slice_image_on_order_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    layer = Image(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)
    assert viewer.dims.order != (1, 0, 2)

    viewer.dims.order = (1, 0, 2)

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data[:, 2, :])


def test_async_slice_image_on_ndisplay_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    layer = Image(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)
    assert viewer.dims.ndisplay != 3

    viewer.dims.ndisplay = 3

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data)


def test_async_slice_multiscale_image_on_pan(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    layer = Image(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 2
    image = vispy_layer.layer
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 4, 5]])

    # Simulate panning to the left by changing the corner pixels in the last
    # dimension, which corresponds to x/columns, then triggering a reload.
    image.corner_pixels = np.array([[0, 0, 0], [0, 4, 3]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data[1][1, 0:4, 0:3])


def test_async_slice_multiscale_image_on_zoom(qtbot, make_napari_viewer, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    layer = Image(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 2
    image = vispy_layer.layer
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 4, 5]])

    # Simulate zooming into the middle of the higher resolution image.
    image._data_level = 0
    image.corner_pixels = np.array([[0, 2, 3], [0, 6, 7]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data[0][2, 2:6, 3:7])


def test_slicing_in_progress(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    layer = LockableImage(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)
    assert viewer.dims.current_step != (2, 0, 0)

    layer = vispy_layer.layer
    assert not viewer.slicing_in_progress
    with layer.lock:
        viewer.dims.current_step = (2, 0, 0)
        assert viewer.slicing_in_progress

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data[2, :, :])
    assert not viewer.slicing_in_progress


def test_async_slice_image_loaded(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    layer = LockableImage(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)

    assert layer.loaded
    assert viewer.dims.current_step != (2, 0, 0)

    with layer.lock:
        viewer.dims.current_step = (2, 0, 0)
        assert not layer.loaded

    wait_until_vispy_image_data_equal(qtbot, vispy_layer, data[2, :, :])
    assert layer.loaded


def test_async_slice_points_loaded(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    shape = (10, 4)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = LockablePoints(data)
    vispy_layer = setup_viewer_for_async_slice(viewer, layer)

    assert layer.loaded
    assert viewer.dims.ndisplay != 3
    with layer.lock:
        viewer.dims.ndisplay = 3
        assert not layer.loaded

    wait_until_vispy_image_data_equal(
        qtbot, vispy_layer, np.flip(data[:, 1:][-3]).reshape(1, 3)
    )
    assert layer.loaded


def setup_viewer_for_async_slice(
    viewer: Viewer,
    layer: Layer,
) -> VispyImageLayer:
    # Initially force synchronous slicing so any slicing caused
    # by adding the image finishes before any other slicing starts.
    viewer._layer_slicer._force_sync = True
    # add layer and get the corresponding vispy image.
    viewer.layers.append(layer)
    vispy_layer = viewer.window._qt_viewer.layer_to_visual[layer]
    # Then allow asynchronous slicing for testing.
    viewer._layer_slicer._force_sync = False
    return vispy_layer


def wait_until_vispy_image_data_equal(
    qtbot, vispy_layer: VispyImageLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_image_data_equal() -> None:
        node = vispy_layer.node
        if isinstance(node, VolumeVisual):
            data = node._last_data
        elif isinstance(node, PointsVisual):
            data = tuple(vertex[0] for vertex in node._subvisuals[0]._data)
        else:
            data = node._data

        # Vispy node data may have been post-processed (e.g. through a colormap),
        # so check that values are close rather than exactly equal.
        np.testing.assert_allclose(data, expected_data)

    qtbot.waitUntil(assert_vispy_image_data_equal)

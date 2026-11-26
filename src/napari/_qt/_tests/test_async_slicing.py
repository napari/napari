# The tests in this module for the new style of async slicing in napari:
# https://napari.org/dev/naps/4-async-slicing.html
import threading
from functools import partial

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QObject, Signal
from vispy.visuals import VolumeVisual

from napari import Viewer
from napari._tests.utils import LockableData
from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.layers.image import VispyImageLayer
from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.layers.vectors import (
    VispyVectorsLayer,
    generate_vector_meshes_2D,
)
from napari.layers import Image, Layer, Points, Vectors
from napari.utils.events import Event, EventEmitter


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def _enable_async(_fresh_settings, make_napari_viewer):
    """
    This fixture depends on _fresh_settings and make_napari_viewer
    to enforce proper order of fixture execution.
    """
    from napari import settings

    settings.get_settings().experimental.async_ = True


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_on_current_step_change(
    make_napari_viewer, qtbot, rng
):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)
    assert viewer.dims.current_step != (2, 0, 0)

    viewer.dims.current_step = (2, 0, 0)

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[2, :, :])


@pytest.mark.usefixtures('_enable_async')
def test_async_out_of_bounds_layer_loaded(make_napari_viewer, qtbot):
    """Check that images that are out of bounds when slicing appear loaded.

    See https://github.com/napari/napari/issues/7070.
    """
    viewer = make_napari_viewer()
    l0 = viewer.add_image(np.random.random((5, 5, 5)))
    l1 = viewer.add_image(np.random.random((5, 5, 5)), translate=(5, 0, 0))
    assert viewer.dims.nsteps == (10, 5, 5)

    def layer_loaded(ly):
        return ly.loaded

    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.current_step = (i, 0, 0)
        qtbot.waitUntil(partial(layer_loaded, l0), timeout=500)
        qtbot.waitUntil(partial(layer_loaded, l1), timeout=500)


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_on_order_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 5, 7))
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)
    assert viewer.dims.order != (1, 0, 2)

    viewer.dims.order = (1, 0, 2)

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[:, 2, :])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_on_ndisplay_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)
    assert viewer.dims.ndisplay != 3

    viewer.dims.ndisplay = 3

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data)


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_multiscale_image_on_pan(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)

    # Force the image to be at the lower resolution, with #7870 this behavior
    # changed so that initial zoom was high, resulting in data_level=0.
    # Likely due to better sequencing of async slicing.
    viewer.scene.camera.zoom = 0.1
    viewer.window._qt_viewer.canvas.on_draw(None)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 1
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 3, 4]])

    # Simulate panning to the left by changing the corner pixels in the last
    # dimension, which corresponds to x/columns, then triggering a reload.
    image.corner_pixels = np.array([[0, 0, 0], [0, 3, 2]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[1][0, 0:4, 0:3])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_multiscale_image_on_zoom(qtbot, make_napari_viewer, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)

    # Force the image to be at the lower resolution, with #7870 this behavior
    # changed so that initial zoom was high, resulting in data_level=0.
    # Likely due to better sequencing of async slicing.
    viewer.scene.camera.zoom = 0.1
    viewer.window._qt_viewer.canvas.on_draw(None)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 1
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 3, 4]])

    # Simulate zooming into the middle of the higher resolution image.
    image._data_level = 0
    image.corner_pixels = np.array([[0, 2, 3], [0, 5, 6]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[0][1, 2:6, 3:7])


@pytest.mark.usefixtures('_enable_async')
def test_async_first_multiscale_slice_is_coarse_and_centered(
    make_napari_viewer, qtbot, monkeypatch
):
    viewer = make_napari_viewer(show=True)
    # Only an initialized canvas repaints synchronously on resize.
    canvas = viewer.window._qt_viewer.canvas._scene_canvas.native
    qtbot.waitUntil(canvas.isValid)

    submitted = []
    submit = viewer._layer_slicer.submit

    def record(*, layers, dims, force=False):
        submitted.append(
            (dims.current_step, [layer.data_level for layer in layers])
        )
        return submit(layers=layers, dims=dims, force=force)

    monkeypatch.setattr(viewer._layer_slicer, 'submit', record)

    data = [np.zeros((8, 256 // 2**i, 256 // 2**i)) for i in range(3)]
    layer = viewer.add_image(data, multiscale=True)
    qtbot.waitUntil(lambda: layer.loaded)

    assert submitted[0] == (viewer.dims.current_step, [2])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_points_on_current_step_change(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.array(
        [
            [0, 2, 3],
            [1, 3, 4],
            [2, 4, 5],
            [3, 5, 6],
            [4, 6, 7],
        ]
    )
    points = Points(data)
    vispy_points = setup_viewer_for_async_slicing(viewer, points)
    assert viewer.dims.current_step != (3, 0, 0)

    viewer.dims.current_step = (3, 0, 0)

    wait_until_vispy_points_data_equal(qtbot, vispy_points, np.array([[5, 6]]))


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_points_on_point_change(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    # Define data so that slicing at 1.6 in the first dimension should match the
    # second point, but won't if that index is prematurely rounded as for other
    # layers.
    data = np.array(
        [
            [0, 2, 3],
            [1.4, 3, 4],
            [2.4, 4, 5],
            [3.4, 5, 6],
            [4, 6, 7],
        ]
    )
    points = Points(data)
    vispy_points = setup_viewer_for_async_slicing(viewer, points)
    assert viewer.dims.point != (1.6, 0, 0)

    viewer.dims.point = (1.6, 0, 0)

    wait_until_vispy_points_data_equal(qtbot, vispy_points, np.array([[3, 4]]))


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_loaded(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    lockable_data = LockableData(data)
    layer = Image(lockable_data, multiscale=False)
    vispy_layer = setup_viewer_for_async_slicing(viewer, layer)

    assert layer.loaded
    assert viewer.dims.current_step != (2, 0, 0)

    with lockable_data.lock:
        viewer.dims.current_step = (2, 0, 0)
        assert not layer.loaded

    qtbot.waitUntil(lambda: layer.loaded)

    np.testing.assert_allclose(vispy_layer.node._data, data[2, :, :])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_vectors_on_current_step_change(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.array(
        [
            [[0, 2, 3], [1, 2, 2]],
            [[2, 4, 5], [0, -3, 3]],
            [[4, 6, 7], [3, 0, -2]],
        ]
    )
    vectors = Vectors(data)
    vispy_vectors = setup_viewer_for_async_slicing(viewer, vectors)
    assert viewer.dims.current_step != (2, 0, 0)

    viewer.dims.current_step = (2, 0, 0)

    wait_until_vispy_vectors_data_equal(
        qtbot, vispy_vectors, np.array([[[2, 4, 5], [0, -3, 3]]])
    )


@pytest.mark.usefixtures('_enable_async')
@pytest.mark.parametrize('backend', ['zarr', 'numpy'])
def test_async_labels_edits_after_slice_read(
    make_napari_viewer, qtbot, backend
):
    viewer = make_napari_viewer()
    shape = (5, 40, 40)
    if backend == 'zarr':
        zarr = pytest.importorskip('zarr')
        data = zarr.zeros(shape=shape, chunks=(1, 40, 40), dtype='int32')
    else:
        data = np.zeros(shape, dtype=np.int32)
    labels = viewer.add_labels(data)
    labels.brush_size = 1
    qtbot.waitUntil(lambda: labels.loaded)

    # Let the worker finish reading the new plane before painting. Qt
    # delivers its response once we return control to the event loop.
    viewer.dims.set_current_step(0, 3)
    viewer._layer_slicer.wait_until_idle(timeout=5)
    assert not labels.loaded
    reloads = []
    labels.events.reload.connect(reloads.append)
    labels.paint((3, 8, 8), 1)
    labels.paint((3, 16, 16), 2, refresh=False)
    labels.data_setitem(tuple(np.array([i]) for i in (3, 24, 24)), 3)

    qtbot.waitUntil(lambda: labels.loaded)
    np.testing.assert_array_equal(labels._slice.image.raw, data[3])
    node = viewer.window._qt_viewer.canvas.layer_to_visual[labels].node
    for i, value in [(8, 1), (16, 2), (24, 3)]:
        assert data[3, i, i] == value
        assert node._data[i, i] != node._data[0, 0]
    assert len(reloads) == (1 if backend == 'zarr' else 0)


@pytest.mark.usefixtures('_enable_async')
def test_async_labels_edits_before_and_after_read(make_napari_viewer, qtbot):
    zarr = pytest.importorskip('zarr')
    viewer = make_napari_viewer()
    shape = (5, 32, 32)
    slow_data = LockableData(np.zeros(shape))
    image = viewer.add_image(slow_data, multiscale=False)
    data = zarr.zeros(shape=shape, chunks=(1, 32, 32), dtype='int32')
    labels = viewer.add_labels(data)
    labels.brush_size = 3
    qtbot.waitUntil(lambda: labels.loaded and image.loaded)

    reloads = []
    # Keep the worker on the image until the indexed edit has been written.
    with slow_data.lock:
        image.refresh()
        viewer.dims.set_current_step(0, 3)
        labels.events.reload.connect(reloads.append)
        labels.data_setitem(tuple(np.array([i]) for i in (3, 2, 2)), 1)
    viewer._layer_slicer.wait_until_idle(timeout=5)
    assert not labels.loaded

    # Qt now holds a copy that predates these overlapping strokes. A stroke
    # on another plane must not leak into the incoming plane.
    labels.paint((3, 10, 10), 2)
    labels.paint((3, 10, 11), 3)
    labels.paint((2, 16, 16), 4)
    qtbot.waitUntil(lambda: labels.loaded)
    assert not reloads
    expected = data[3]
    assert expected[2, 2] == 1
    assert expected[10, 11] == 3
    assert expected[16, 16] == 0
    np.testing.assert_array_equal(labels._slice.image.raw, expected)
    node = viewer.window._qt_viewer.canvas.layer_to_visual[labels].node
    np.testing.assert_array_equal(
        node._data, labels._raw_to_displayed(expected)
    )


@pytest.mark.parametrize('next_request', ['scroll', 'refresh'])
@pytest.mark.usefixtures('_enable_async')
def test_async_labels_stroke_on_plane_delivered_after_newer_request(
    make_napari_viewer, qtbot, monkeypatch, next_request
):
    from napari.layers.labels._slice import _LabelsSliceRequest

    zarr = pytest.importorskip('zarr')
    viewer = make_napari_viewer()
    data = zarr.zeros(shape=(5, 32, 32), chunks=(1, 32, 32), dtype='int32')
    labels = viewer.add_labels(data)
    labels.brush_size = 3
    qtbot.waitUntil(lambda: labels.loaded)

    copied = threading.Event()
    deliver = threading.Event()
    read = _LabelsSliceRequest.__call__

    def hold_first_response(request):
        response = read(request)
        if not copied.is_set():
            copied.set()
            deliver.wait(5)
        return response

    monkeypatch.setattr(_LabelsSliceRequest, '__call__', hold_first_response)
    viewer.dims.set_current_step(0, 3)
    assert copied.wait(5)
    # Plane 3 was copied before this stroke, and a newer request follows
    # before the copy is delivered.
    labels.paint((3, 10, 10), 7)
    if next_request == 'scroll':
        viewer.dims.set_current_step(0, 4)
    else:
        labels.refresh()

    shown = []
    update = labels._slicing_state._update_slice_response

    def record_plane_3(response):
        update(response)
        if response.slice_input.world_slice.point[0] == 3:
            shown.append(int(labels._slice.image.raw[10, 10]))

    monkeypatch.setattr(
        labels._slicing_state, '_update_slice_response', record_plane_3
    )
    deliver.set()
    qtbot.waitUntil(lambda: labels.loaded)
    assert data[3, 10, 10] == 7
    # Without showing older reads, plane 3 is not delivered after scrolling.
    assert shown or next_request == 'scroll'
    assert all(value == 7 for value in shown)
    assert not labels._slicing_state._pending_slices


@pytest.mark.usefixtures('_enable_async')
@pytest.mark.parametrize(('brush_size', 'strokes'), [(151, 1), (1, 33)])
def test_async_labels_repair_limit(
    make_napari_viewer, qtbot, brush_size, strokes
):
    zarr = pytest.importorskip('zarr')
    viewer = make_napari_viewer()
    data = zarr.zeros(shape=(3, 192, 192), chunks=(1, 64, 64), dtype='int32')
    labels = viewer.add_labels(data)
    labels.brush_size = brush_size
    qtbot.waitUntil(lambda: labels.loaded)
    labels.refresh()
    viewer._layer_slicer.wait_until_idle(timeout=5)

    reloads = []
    labels.events.reload.connect(reloads.append)
    for i in range(strokes):
        labels.paint((1, 96, 96 + i), 1)
    qtbot.waitUntil(lambda: labels.loaded)
    assert len(reloads) == 1
    assert data[1, 96, 96] == 1
    np.testing.assert_array_equal(labels._slice.image.raw, data[1])


@pytest.mark.usefixtures('_enable_async')
@pytest.mark.parametrize('backend', ['numpy', 'zarr'])
def test_async_labels_paint_during_slice_delivery(
    make_napari_viewer, qtbot, backend
):
    viewer = make_napari_viewer()
    if backend == 'zarr':
        data = pytest.importorskip('zarr').zeros(
            shape=(3, 20, 20), dtype='int32'
        )
    else:
        data = np.zeros((3, 20, 20), dtype=np.int32)
    labels = viewer.add_labels(data)
    qtbot.waitUntil(lambda: labels.loaded)
    labels.brush_size = 1

    def paint_once(event):
        labels._transforms.events.changed.disconnect(paint_once)
        labels.paint((1, 5, 5), 1)

    # A transform callback can paint after the incoming raw plane has been
    # converted into a separate texture array but before the layer is loaded.
    labels._transforms.events.changed.connect(paint_once)
    labels.refresh()
    qtbot.waitUntil(lambda: labels.loaded)
    node = viewer.window._qt_viewer.canvas.layer_to_visual[labels].node
    assert labels.data[1, 5, 5] == 1
    assert node._data[5, 5] != node._data[0, 0]


@pytest.mark.usefixtures('_enable_async')
def test_async_labels_batch_spans_slice_delivery(make_napari_viewer, qtbot):
    zarr = pytest.importorskip('zarr')
    viewer = make_napari_viewer()
    data = zarr.zeros(shape=(20, 20), dtype='int32')
    labels = viewer.add_labels(data)
    labels.brush_size = 1
    viewer._layer_slicer.wait_until_idle(timeout=5)
    assert not labels.loaded

    # The first slice is still waiting for Qt. Indexing the placeholder
    # here would raise, and updating only the second pixel would lose the first.
    labels.data_setitem((np.array([3]), np.array([3])), 1, refresh=False)
    qtbot.waitUntil(lambda: labels.loaded)
    labels.paint((10, 10), 2)
    np.testing.assert_array_equal(labels._slice.image.raw, data[:])
    node = viewer.window._qt_viewer.canvas.layer_to_visual[labels].node
    assert node._data[3, 3] != node._data[0, 0]
    assert node._data[10, 10] != node._data[0, 0]


@pytest.mark.usefixtures('_enable_async')
def test_async_labels_edit_then_scroll(make_napari_viewer, qtbot):
    zarr = pytest.importorskip('zarr')
    viewer = make_napari_viewer()
    data = zarr.zeros(shape=(5, 20, 20), chunks=(1, 20, 20), dtype='int32')
    labels = viewer.add_labels(data)
    qtbot.waitUntil(lambda: labels.loaded)
    labels.refresh()
    viewer._layer_slicer.wait_until_idle(timeout=5)
    reloads = []
    labels.events.reload.connect(reloads.append)
    labels.paint((2, 10, 10), 1)
    viewer.dims.set_current_step(0, 3)
    qtbot.waitUntil(lambda: labels.loaded)
    np.testing.assert_array_equal(labels._slice.image.raw, data[3])
    viewer.dims.set_current_step(0, 2)
    qtbot.waitUntil(lambda: labels.loaded)
    assert labels._slice.image.raw[10, 10] == 1
    assert not reloads


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_two_layers_shutdown(make_napari_viewer):
    """See https://github.com/napari/napari/issues/6685"""
    viewer = make_napari_viewer()
    # To reproduce the issue, we need two points layers where the second has
    # some non-zero coordinates.
    viewer.add_points()
    points = viewer.add_points()
    points.add([[1, 2]])

    viewer.close()


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_response_applied_on_main_thread(
    make_napari_viewer, qtbot, rng, monkeypatch
):
    viewer = make_napari_viewer()
    image = Image(rng.random((3, 4, 5)))
    setup_viewer_for_async_slicing(viewer, image)

    ready_threads = []
    viewer._layer_slicer.events.ready.connect(
        lambda _: ready_threads.append(threading.current_thread())
    )
    update_threads = []
    update_slice_response = image._slicing_state._update_slice_response

    def record_update_thread(response):
        update_threads.append(threading.current_thread())
        update_slice_response(response)

    monkeypatch.setattr(
        image._slicing_state, '_update_slice_response', record_update_thread
    )

    viewer.dims.current_step = (2, 0, 0)

    qtbot.waitUntil(lambda: bool(update_threads))
    assert threading.main_thread() not in ready_threads
    assert set(update_threads) == {threading.main_thread()}


def test_callback_emitting_signal_of_deleted_qobject_is_disconnected(qapp):
    class Source(QObject):
        signal = Signal()

        def emit_signal(self):
            self.signal.emit()

    source = Source()
    emitter = EventEmitter(type_name='test')
    emitter.connect(source.emit_signal)

    source.deleteLater()
    QCoreApplication.sendPostedEvents(source, QEvent.Type.DeferredDelete)
    emitter()

    assert not emitter.callbacks


def setup_viewer_for_async_slicing(
    viewer: Viewer,
    layer: Layer,
) -> VispyBaseLayer:
    # Initially force synchronous slicing so any slicing caused
    # by adding the layer finishes before any other slicing starts.
    with viewer._layer_slicer.force_sync():
        # Add the layer and get the corresponding vispy layer.
        layer = viewer.add_layer(layer)
        vispy_layer = viewer.window._qt_viewer.layer_to_visual[layer]

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


def wait_until_vispy_points_data_equal(
    qtbot, vispy_layer: VispyPointsLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_points_data_equal() -> None:
        positions = vispy_layer.node._subvisuals[0]._data['a_position']
        # Flip the coordinates because vispy uses xy instead of rc ordering.
        # Also only take the number of dimensions expected since vispy points
        # are always 3D even when displaying 2D slices.
        data = positions[:, -expected_data.shape[1] :: -1]
        np.testing.assert_array_equal(data, expected_data)

    qtbot.waitUntil(assert_vispy_points_data_equal)


def wait_until_vispy_vectors_data_equal(
    qtbot, vispy_layer: VispyVectorsLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_vectors_data_equal() -> None:
        displayed = expected_data[..., -2:]
        exp_vertices, exp_faces = generate_vector_meshes_2D(
            displayed, 1, 1, 'triangle'
        )
        meshdata = vispy_layer.node._meshdata
        vertices = meshdata.get_vertices()
        faces = meshdata.get_faces()
        # invert for vispy
        np.testing.assert_array_equal(vertices, exp_vertices[..., ::-1])
        np.testing.assert_array_equal(faces, exp_faces)

    qtbot.waitUntil(assert_vispy_vectors_data_equal)

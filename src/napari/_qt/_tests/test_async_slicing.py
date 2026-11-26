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
from napari.utils.notifications import notification_manager


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


class ReadRecordingData(LockableData):
    """LockableData that records the first index of each read.

    A read is recorded before it may block on the lock.
    """

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        self.read_planes: list[int] = []

    def __getitem__(self, key):
        index = key[0]
        self.read_planes.append(
            index.start if isinstance(index, slice) else int(index)
        )
        return super().__getitem__(key)


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_superseded_requests_do_not_read_data(
    make_napari_viewer, qtbot, rng
):
    """Superseded slice requests don't read data, and the newest slice of
    every layer is shown.

    Zooming a multiscale image refreshes each layer on its own, while a slice
    change slices all layers together. Interleaving them while the slicing
    thread is busy used to queue up requests that each read their plane.
    """
    viewer = make_napari_viewer()
    datas = [ReadRecordingData(rng.random((8, 4, 5))) for _ in range(2)]
    layers = [Image(data, multiscale=False) for data in datas]
    vispy_layers = [
        setup_viewer_for_async_slicing(viewer, layer) for layer in layers
    ]
    viewer._layer_slicer.wait_until_idle(timeout=5)
    qtbot.waitUntil(lambda: all(layer.loaded for layer in layers))
    for data in datas:
        data.read_planes.clear()

    with datas[0].lock:
        viewer.dims.current_step = (1, 0, 0)
        # Wait until the slicing thread is reading the first layer.
        qtbot.waitUntil(lambda: datas[0].read_planes == [1])
        for z in (5, 6, 7):
            for layer in layers:
                layer.refresh()
            viewer.dims.current_step = (z, 0, 0)

    qtbot.waitUntil(lambda: all(layer.loaded for layer in layers))
    # The read of plane 1 had already started, so only plane 7 is read again.
    assert datas[0].read_planes == [1, 7]
    assert datas[1].read_planes == [7]
    for data, vispy_layer in zip(datas, vispy_layers, strict=True):
        np.testing.assert_allclose(vispy_layer.node._data, data.data[7])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_from_before_points_removal_not_applied(
    make_napari_viewer, qtbot
):
    """See https://github.com/napari/napari/issues/9046"""
    viewer = make_napari_viewer()
    points = viewer.add_points([[10, 10], [20, 20], [30, 30]])

    exceptions = edit_while_slice_is_pending(
        viewer, qtbot, points, lambda: points.remove([2])
    )

    assert not exceptions
    np.testing.assert_array_equal(points._view_indices, [0, 1])


@pytest.mark.usefixtures('_enable_async')
def test_async_points_view_updated_on_removal_before_slicing(
    make_napari_viewer, qtbot
):
    """Removing points updates the view before the new slice is done.

    Until then, hovering over the canvas reads the view.
    See https://github.com/napari/napari/issues/9046
    """
    viewer = make_napari_viewer()
    data = LockableData(np.zeros((5, 40, 40)))
    image = viewer.add_image(
        data, contrast_limits=(0, 1), scale=(1000,) * 3, units=('nm',) * 3
    )
    points = viewer.add_points(
        [[2, 10, 10], [2, 20, 20], [2, 30, 30]], units=('um',) * 3
    )
    viewer.dims.set_point(0, 2000)
    qtbot.waitUntil(lambda: image.loaded and points.loaded)

    # Observers may hit-test as soon as the edit starts a refresh.
    observed = []

    def read_points(event):
        observed.append(points.get_value((2, 20, 20), world=True))

    points.events.loaded.connect(read_points)
    points.events.set_data.connect(read_points)
    with data.lock:
        image.refresh()
        points.selected_data = {2}
        points.remove_selected()

        assert points.loaded
        assert observed == [1]
        np.testing.assert_array_equal(points._view_indices, [0, 1])
        assert points.get_value((2, 20, 20), world=True) == 1

    qtbot.waitUntil(lambda: image.loaded and points.loaded)
    np.testing.assert_array_equal(points._view_indices, [0, 1])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_from_before_fewer_vectors_not_applied(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    data = np.zeros((5, 2, 2))
    data[:, 0] = np.arange(5)[:, np.newaxis] * 5
    data[:, 1] = 1
    vectors = viewer.add_vectors(data)

    def edit():
        vectors.data = data[:2]
        # The view is updated before the new slice is done.
        np.testing.assert_array_equal(vectors._view_indices, [0, 1])
        assert len(vectors._view_face_color) == 2

    exceptions = edit_while_slice_is_pending(viewer, qtbot, vectors, edit)

    assert not exceptions
    np.testing.assert_array_equal(vectors._view_indices, [0, 1])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_from_before_image_ndim_change_not_applied(
    make_napari_viewer, qtbot, rng
):
    viewer = make_napari_viewer()
    image = viewer.add_image(rng.random((5, 40, 40)))
    new_data = rng.random((30, 30))

    exceptions = edit_while_slice_is_pending(
        viewer, qtbot, image, lambda: setattr(image, 'data', new_data)
    )

    assert not exceptions
    np.testing.assert_array_equal(image._slice.image.raw, new_data)


@pytest.mark.usefixtures('_enable_async')
def test_async_point_text_listener_during_removal(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    points = viewer.add_points(
        [[10, 10], [20, 20], [30, 30]],
        features={'label': ['a', 'b', 'c']},
        text='{label}',
    )
    observed = []

    def read_points(event):
        QCoreApplication.processEvents()
        observed.append(points.get_value((20, 20)))

    points.text.events.values.connect(read_points)
    exceptions = edit_while_slice_is_pending(
        viewer, qtbot, points, lambda: points.remove([2])
    )
    assert not exceptions
    assert observed == [1]


@pytest.mark.usefixtures('_enable_async')
def test_async_points_paste_discards_completed_slice(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    points = viewer.add_points([[10, 10], [20, 20]])
    qtbot.waitUntil(lambda: points.loaded)
    points.selected_data = {0}
    points._copy_data()

    displayed = []
    points.events.set_data.connect(
        lambda: displayed.append(points._view_indices.tolist())
    )
    exceptions = edit_while_slice_is_pending(
        viewer, qtbot, points, points._paste_data
    )

    assert not exceptions
    assert displayed
    assert all(indices == [0, 1, 2] for indices in displayed)


@pytest.mark.usefixtures('_enable_async')
def test_async_edit_while_navigating(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = LockableData(np.zeros((5, 40, 40)))
    image = viewer.add_image(data, contrast_limits=(0, 1))
    points = viewer.add_points([[2, 10, 10], [2, 20, 20], [3, 30, 30]])
    qtbot.waitUntil(lambda: image.loaded and points.loaded)
    with data.lock:
        viewer.dims.set_current_step(0, 3)
        points.remove([0])
        np.testing.assert_array_equal(points._view_indices, [0])
        assert not points.loaded
    qtbot.waitUntil(lambda: image.loaded and points.loaded)
    np.testing.assert_array_equal(points._view_indices, [1])


@pytest.mark.usefixtures('_enable_async')
def test_async_edit_hidden_image_with_pending_slice(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((4, 5)), contrast_limits=(0, 1))
    new_data = np.ones((4, 5))
    updates = []

    def replace_hidden_data():
        image.visible = False
        image.data = new_data
        image.events.set_data.connect(
            lambda: updates.append(image._slice.image.raw)
        )

    exceptions = edit_while_slice_is_pending(
        viewer, qtbot, image, replace_hidden_data
    )
    assert not exceptions
    assert not updates
    image.visible = True
    qtbot.waitUntil(lambda: image.loaded)
    np.testing.assert_array_equal(image._slice.image.raw, new_data)


@pytest.mark.usefixtures('_enable_async')
def test_async_deliver_intermediate_plane_for_current_data(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    data = LockableData(np.arange(3 * 4 * 5).reshape(3, 4, 5))
    image = viewer.add_image(data, contrast_limits=(0, 60))
    qtbot.waitUntil(lambda: image.loaded)
    viewer.dims.set_current_step(0, 0)
    viewer._layer_slicer.wait_until_idle(timeout=5)
    with data.lock:
        viewer.dims.set_current_step(0, 2)
        qtbot.waitUntil(
            lambda: np.array_equal(image._slice.image.raw, data.data[0])
        )
        assert not image.loaded
    qtbot.waitUntil(lambda: image.loaded)
    np.testing.assert_array_equal(image._slice.image.raw, data.data[2])


@pytest.mark.usefixtures('_enable_async')
def test_transform_listener_delivering_newer_plane(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    volume = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    image = viewer.add_image(volume, contrast_limits=(0, 60))
    qtbot.waitUntil(lambda: image.loaded)

    def slice_at(z):
        dims = viewer.dims.model_copy(update={'point': (z, 0, 0)})
        (future,) = viewer._layer_slicer.submit(layers=[image], dims=dims)
        return future.result(5)

    first = slice_at(0)
    second = slice_at(2)
    deliver = viewer.window._qt_viewer._on_slice_ready

    def deliver_second(event):
        image._transforms.events.changed.disconnect(deliver_second)
        np.testing.assert_array_equal(image._slice.image.raw, volume[0])
        deliver(second)

    published = []
    image.events.set_data.connect(
        lambda: published.append(image._slice.request_id)
    )
    image._transforms.events.changed.connect(deliver_second)
    deliver(first)
    QCoreApplication.processEvents()
    assert published == [image._slice.request_id]
    np.testing.assert_array_equal(image._slice.image.raw, volume[2])
    assert image._slice_input.world_slice.point[0] == 2
    assert image.loaded


@pytest.mark.usefixtures('_enable_async')
def test_pending_plane_after_axis_reorder(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = LockableData(np.arange(60).reshape(3, 4, 5))
    image = viewer.add_image(data, contrast_limits=(0, 60))
    qtbot.waitUntil(lambda: image.loaded)
    viewer.dims.set_current_step(0, 0)
    viewer._layer_slicer.wait_until_idle(5)
    updates = []
    image.events.set_data.connect(
        lambda: updates.append(image._slice_input.order)
    )
    with data.lock:
        viewer.dims.order = (1, 0, 2)
        QCoreApplication.processEvents()
        assert not updates
    qtbot.waitUntil(lambda: image.loaded)
    assert updates
    assert all(order == (1, 0, 2) for order in updates)


@pytest.mark.usefixtures('_enable_async')
@pytest.mark.parametrize('newer', ['sync', 'remove'])
def test_async_late_plane_cannot_replace_current_view(
    make_napari_viewer, qtbot, monkeypatch, newer
):
    viewer = make_napari_viewer()
    volume = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    image = Image(volume, contrast_limits=(0, 60))
    visual = setup_viewer_for_async_slicing(viewer, image)
    slicer = viewer._layer_slicer

    # Let the older request finish only after the newer plane is on screen.
    # Navigate towards decreasing Z, where comparing plane numbers is wrong.
    started = threading.Event()
    release = threading.Event()
    make_request = image._slicing_state._make_slice_request

    def make_blocked_request(dims):
        request = make_request(dims)
        if dims.current_step[0] != 2:
            return request

        def read():
            started.set()
            assert release.wait(10)
            return request()

        read.id = request.id
        return read

    monkeypatch.setattr(
        image._slicing_state, '_make_slice_request', make_blocked_request
    )
    updates = []
    image.events.set_data.connect(
        lambda: updates.append(image._slice.image.raw)
    )
    try:
        old_dims = viewer.dims.model_copy(update={'point': (2, 0, 0)})
        slicer.submit(layers=[image], dims=old_dims)
        assert started.wait(10)
        if newer == 'remove':
            viewer.layers.remove(image)
        else:
            with slicer.force_sync():
                viewer.dims.set_current_step(0, 0)
            wait_until_vispy_image_data_equal(qtbot, visual, volume[0])
            assert image.loaded
        updates.clear()
    finally:
        release.set()
    slicer.wait_until_idle(10)
    QCoreApplication.processEvents()
    assert not updates
    assert image.loaded
    if newer != 'remove':
        np.testing.assert_array_equal(image._slice.image.raw, volume[0])
        np.testing.assert_array_equal(visual.node._data, volume[0])


@pytest.mark.usefixtures('_enable_async')
def test_layer_presented_while_sibling_is_reading(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    fast = Image(data, contrast_limits=(0, 60))
    slow_data = LockableData(data + 100)
    slow = Image(slow_data, contrast_limits=(100, 160))
    fast_visual = setup_viewer_for_async_slicing(viewer, fast)
    slow_visual = setup_viewer_for_async_slicing(viewer, slow)
    dims = viewer.dims.model_copy(update={'point': (2, 0, 0)})

    with slow_data.lock:
        viewer._layer_slicer.submit(layers=[fast, slow], dims=dims)
        wait_until_vispy_image_data_equal(qtbot, fast_visual, data[2])
        assert fast.loaded
        assert not slow.loaded
    wait_until_vispy_image_data_equal(qtbot, slow_visual, data[2] + 100)
    assert slow.loaded


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_failure_updates_loaded_on_main_thread(
    make_napari_viewer, qtbot, rng, monkeypatch
):
    viewer = make_napari_viewer()
    image = Image(rng.random((3, 4, 5)))
    setup_viewer_for_async_slicing(viewer, image)

    errors = []
    loaded_threads = []
    make_request = image._slicing_state._make_slice_request

    def record_error(exctype, error, traceback):
        errors.append(error)

    def update_loaded():
        if image.loaded:
            loaded_threads.append(threading.current_thread())

    def failing_request(dims):
        class Request:
            id = make_request(dims).id

            def __call__(self):
                raise RuntimeError('failed to read slice')

        return Request()

    monkeypatch.setattr(notification_manager, 'receive_error', record_error)
    monkeypatch.setattr(
        image._slicing_state, '_make_slice_request', failing_request
    )
    image.events.loaded.connect(update_loaded)

    viewer._layer_slicer.submit(layers=[image], dims=viewer.dims)
    assert not image.loaded
    qtbot.waitUntil(lambda: image.loaded and bool(errors))

    assert len(errors) == 1
    assert str(errors[0]) == 'failed to read slice'
    assert loaded_threads == [threading.main_thread()]


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


def edit_while_slice_is_pending(viewer, qtbot, layer, edit) -> list:
    """Slice the layer, and edit it after the slice is done but before it is
    applied. Returns the exceptions raised until the layer is loaded again.

    The slicing thread finishes the slice right away, but the main thread
    only applies it when it next processes events, which is after `edit`.
    """
    qtbot.waitUntil(lambda: layer.loaded)
    layer.refresh()
    viewer._layer_slicer.wait_until_idle(timeout=5)
    assert not layer.loaded

    with qtbot.capture_exceptions() as exceptions:
        edit()
        qtbot.waitUntil(lambda: layer.loaded)
    return exceptions


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

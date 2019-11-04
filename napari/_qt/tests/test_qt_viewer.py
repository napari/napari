import numpy as np
import pytest

from napari.components import ViewerModel
from napari._qt.qt_viewer import QtViewer


def test_qt_viewer(qtbot):
    """Test instantiating viewer."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_image(qtbot):
    """Test adding image."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_volume(qtbot):
    """Test adding volume."""
    viewer = ViewerModel(ndisplay=3)
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_image(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_pyramid(qtbot):
    """Test adding image pyramid."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_image(data, is_pyramid=True)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_labels(qtbot):
    """Test adding labels image."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_points(qtbot):
    """Test adding points."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_vectors(qtbot):
    """Test adding vectors."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_add_shapes(qtbot):
    """Test adding vectors."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_labels(qtbot):
    """Test adding new labels layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    viewer._new_labels()
    assert np.max(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add labels with image already present
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_labels()
    assert np.max(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_points(qtbot):
    """Test adding new points layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    viewer.add_points()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_points()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_new_shapes(qtbot):
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    viewer.add_shapes()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Add points with image already present
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_shapes()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0


def test_screenshot(qtbot):
    "Test taking a screenshot"
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot
    screenshot = view.screenshot()
    assert screenshot.ndim == 3


@pytest.mark.parametrize(
    "dtype", ['int8', 'uint8', 'int16', 'uint16', 'float32']
)
def test_qt_viewer_data_integrity(qtbot, dtype):
    """Test that the viewer doesn't change the underlying array."""

    image = np.random.rand(10, 32, 32)
    image *= 200 if dtype.endswith('8') else 2 ** 14
    image = image.astype(dtype)
    imean = image.mean()

    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    viewer.add_image(image.copy())
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
    # toggle dimensions
    viewer.dims.ndisplay = 3
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean
    # back to 2D
    viewer.dims.ndisplay = 2
    datamean = viewer.layers[0].data.mean()
    assert datamean == imean


@pytest.fixture()
def view(qtbot):
    """A fixture to handle timing associated with axis animations"""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 10, 15))
    viewer.add_image(data)
    view.dims._frame = 0
    axis = 0

    def increment(e):
        view.dims._frame = e.value  # this is provided by the axis event
        # if we don't "enable play" again, view.dims won't request a new frame
        view.dims._play_ready = True

    def start_and_wait(interval, nframes, frame_range=None, mode='loop'):
        view.dims._frame = 0
        view.dims.play(
            axis, 1000 / interval, frame_range=frame_range, playback_mode=mode
        )
        # wait for the thread to start before timing...
        qtbot.waitSignal(view.dims._animation_thread.started, timeout=10000)
        qtbot.wait(abs(interval) * (nframes + 0.5))
        view.dims.stop()
        return view.dims._frame

    view.dims.dims.events.axis.connect(increment)
    view.start_and_wait = start_and_wait
    return view


def test_play_forward(qtbot, view):
    """Test that play changes the slice on axis 0."""
    interval, nframes = 50, 5
    endframe = view.start_and_wait(interval, nframes)
    assert nframes <= endframe <= nframes + 2
    # also make sure that the stop button worked and killed the animation
    # thread
    qtbot.wait(100)
    assert view.dims._frame == endframe
    assert not hasattr(view.dims, '_animation_thread')

    with pytest.raises(IndexError):
        view.dims.play(4, 20)
        qtbot.wait(20)
        view.dims.stop()


def test_play_backward(qtbot, view):
    """Test that play works with a negative fps."""
    interval, nframes = 50, 5
    nz = view.dims.dims.range[0][1]
    endframe = view.start_and_wait(-interval, nframes)
    assert nz - nframes - 2 <= endframe <= nz - nframes + 1


def test_play_with_range(qtbot, view):
    """Test that play works with frame_range."""
    interval, nframes = 50, 5
    _range = [2, 9]
    endframe = view.start_and_wait(interval, nframes, frame_range=_range)
    assert endframe >= _range[0] + nframes

    with pytest.raises(ValueError):
        # frame_range[1] not > frame_range[0]
        view.dims.play(0, 20, frame_range=[2, 2])
        qtbot.wait(20)
        view.dims.stop()

    with pytest.raises(IndexError):
        # data doesn't have 20 frames
        view.dims.play(0, 20, frame_range=[2, 20])
        qtbot.wait(20)
        view.dims.stop()


@pytest.mark.parametrize("mode", ['loop', 'loop_back_and_forth', 'once'])
def test_play_with_loops(qtbot, view, mode):
    """Test that the various play playback_modes work."""
    interval = 50
    nz = view.dims.dims.range[0][1]
    extra = 2
    endframe = view.start_and_wait(interval, nz + extra, mode=mode)

    if mode == 'once':
        # should have stopped at last frame
        assert nz - 1 <= endframe <= nz
    elif mode == 'loop':
        # should have wrapped around to first frame
        assert extra - 1 <= endframe <= extra + 2
    else:
        # should have looped back and forth
        assert nz - extra - 1 >= endframe >= nz - extra - 3


def test_play_with_loops_fails(qtbot, view):
    """Test that playback_mode only accepts known parameters"""

    with pytest.raises(ValueError):
        view.dims.play(0, 20, playback_mode='enso')
        qtbot.wait(20)
        view.dims.stop()

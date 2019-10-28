import numpy as np
import pytest
from unittest.mock import patch

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


def test_play_axis(qtbot):
    """Test that play_axis changes the slice on axis 0."""

    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    nz = 6
    data = np.random.random((nz, 10, 15))
    viewer.add_image(data)

    dims_model = view.dims.dims

    def fake_draw(axis, frame):
        # make sure that we never request a frame outside of the bounds of the data
        assert frame < nz
        # mocks the dims.set_point() method
        dims_model._point[axis] = frame
        # simulates a vispy draw cascade
        # then resets the view.dims._play_ready attribute
        view.dims._play_ready = True

    @patch.object(
        dims_model,
        'set_point',
        wraps=dims_model.set_point,
        side_effect=fake_draw,
    )
    def test_play(mock_method, nframes=2, axis=0, interval=100):
        # reset the current slice position to 0
        dims_model._point[axis] = 0
        assert dims_model.point[axis] == 0
        # play for a litte bit
        view.dims.play_dim(axis, 1000 / interval)
        # the 0.5 allows for generous clock jitter...
        qtbot.wait(interval * (nframes + 0.5))
        view.dims.stop()
        assert mock_method.call_count == nframes + 1

    test_play(nframes=nz - 2)
    # make sure it rolls over ok
    test_play(nframes=nz + 2)

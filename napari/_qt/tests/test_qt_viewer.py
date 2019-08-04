import numpy as np

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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


def test_add_volume(qtbot):
    """Test adding volume."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_volume(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == 1
    assert np.sum(view.dims._displayed) == 0


def test_add_pyramid(qtbot):
    """Test adding image pyramid."""
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_pyramid(data)
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0

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
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


def test_new_points(qtbot):
    """Test adding new points layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    viewer._new_points()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0

    # Add points with image already present
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_points()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


def test_new_shapes(qtbot):
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    viewer._new_shapes()
    assert len(viewer.layers[0].data) == 0
    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0

    # Add points with image already present
    viewer = ViewerModel()
    view = QtViewer(viewer)
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_shapes()
    assert len(viewer.layers[1].data) == 0
    assert len(viewer.layers) == 2
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == 0
    assert np.sum(view.dims._displayed) == 0


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

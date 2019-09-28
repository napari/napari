import numpy as np

from napari import Viewer


def test_viewer(qtbot):
    """Test instantiating viewer."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Close the viewer
    viewer.window.close()


def test_add_image(qtbot):
    """Test adding image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
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

    # Close the viewer
    viewer.window.close()


def test_add_volume(qtbot):
    """Test adding volume."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_image(data)
    viewer.dims.ndisplay = 3
    assert np.all(viewer.layers[0].data == data)

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Close the viewer
    viewer.window.close()


def test_add_pyramid(qtbot):
    """Test adding image pyramid."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
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

    # Close the viewer
    viewer.window.close()


def test_add_labels(qtbot):
    """Test adding labels image."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
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

    # Close the viewer
    viewer.window.close()


def test_add_points(qtbot):
    """Test adding points."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
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

    # Close the viewer
    viewer.window.close()


def test_add_vectors(qtbot):
    """Test adding vectors."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
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

    # Close the viewer
    viewer.window.close()


def test_add_shapes(qtbot):
    """Test adding shapes."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
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

    # Close the viewer
    viewer.window.close()


def test_add_surface(qtbot):
    """Test adding 3D surface."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    viewer.add_surface(data)
    assert np.all(
        [np.all(vd == d) for vd, d in zip(viewer.layers[0].data, data)]
    )

    assert len(viewer.layers) == 1
    assert view.layers.vbox_layout.count() == 2 * len(viewer.layers) + 2

    assert viewer.dims.ndim == 3
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 1

    # Close the viewer
    viewer.window.close()


def test_screenshot(qtbot):
    "Test taking a screenshot"
    viewer = Viewer()
    view = viewer.window.qt_viewer
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
    screenshot = viewer.screenshot()
    assert screenshot.ndim == 3

    # Close the viewer
    viewer.window.close()

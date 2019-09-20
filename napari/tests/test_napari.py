import numpy as np
import napari


def test_add_image(qtbot):
    """Test adding image."""

    np.random.seed(0)
    data = np.random.random((10, 15))

    viewer = napari.view_image(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

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
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer = napari.view_image(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)
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
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer = napari.view_pyramid(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

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
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    viewer = napari.view_labels(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

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
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    viewer = napari.view_points(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

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
    np.random.seed(0)
    data = 20 * np.random.random((10, 2, 2))
    viewer = napari.view_vectors(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)
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
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    viewer = napari.view_shapes(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

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
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    viewer = napari.view_surface(data)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

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

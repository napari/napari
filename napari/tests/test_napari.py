import numpy as np
import napari


def test_view_image(qtbot):
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


def test_view_multichannel(qtbot):
    """Test adding image."""

    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    viewer = napari.view_image(data, channel_axis=-1)
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert np.all(viewer.layers[i].data == data.take(i, axis=-1))

    # Close the viewer
    viewer.window.close()


def test_view_pyramid(qtbot):
    """Test adding image pyramid."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer = napari.view_image(data, is_pyramid=True)
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


def test_view_labels(qtbot):
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


def test_view_points(qtbot):
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


def test_view_vectors(qtbot):
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


def test_view_shapes(qtbot):
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


def test_view_surface(qtbot):
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

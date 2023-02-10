import numpy as np


def test_camera(make_napari_viewer):
    """Test vispy camera creation in 2D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.vispy_camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2

    np.testing.assert_almost_equal(viewer.camera.angles, (0, 0, 90))
    np.testing.assert_almost_equal(viewer.camera.center, (0, 5.0, 5.0))
    np.testing.assert_almost_equal(viewer.camera.angles, vispy_camera.angles)
    np.testing.assert_almost_equal(viewer.camera.center, vispy_camera.center)
    np.testing.assert_almost_equal(viewer.camera.zoom, vispy_camera.zoom)


def test_vispy_camera_update_from_model(make_napari_viewer):
    """Test vispy camera update from model in 2D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.vispy_camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2

    # Update camera center and zoom
    viewer.camera.center = (11, 12)
    viewer.camera.zoom = 4

    np.testing.assert_almost_equal(viewer.camera.angles, (0, 0, 90))
    np.testing.assert_almost_equal(viewer.camera.center, (0, 11, 12))
    np.testing.assert_almost_equal(viewer.camera.zoom, 4)
    np.testing.assert_almost_equal(viewer.camera.angles, vispy_camera.angles)
    np.testing.assert_almost_equal(viewer.camera.center, vispy_camera.center)
    np.testing.assert_almost_equal(viewer.camera.zoom, vispy_camera.zoom)


def test_camera_model_update_from_vispy(make_napari_viewer):
    """Test camera model updates from vispy in 2D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.vispy_camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2

    # Update vispy camera center and zoom
    vispy_camera.center = (11, 12)
    vispy_camera.zoom = 4
    vispy_camera.on_draw(None)

    np.testing.assert_almost_equal(viewer.camera.angles, (0, 0, 90))
    np.testing.assert_almost_equal(viewer.camera.center, (0, 11, 12))
    np.testing.assert_almost_equal(viewer.camera.zoom, 4)
    np.testing.assert_almost_equal(viewer.camera.angles, vispy_camera.angles)
    np.testing.assert_almost_equal(viewer.camera.center, vispy_camera.center)
    np.testing.assert_almost_equal(viewer.camera.zoom, vispy_camera.zoom)


def test_3D_camera(make_napari_viewer):
    """Test vispy camera creation in 3D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.vispy_camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    viewer.dims.ndisplay = 3

    # Test camera values have updated
    np.testing.assert_almost_equal(viewer.camera.angles, (0, 0, 90))
    np.testing.assert_almost_equal(viewer.camera.center, (5.0, 5.0, 5.0))
    np.testing.assert_almost_equal(viewer.camera.angles, vispy_camera.angles)
    np.testing.assert_almost_equal(viewer.camera.center, vispy_camera.center)
    np.testing.assert_almost_equal(viewer.camera.zoom, vispy_camera.zoom)


def test_vispy_camera_update_from_model_3D(make_napari_viewer):
    """Test vispy camera update from model in 3D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.vispy_camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    viewer.dims.ndisplay = 3

    # Update camera angles, center, and zoom
    viewer.camera.angles = (24, 12, -19)
    viewer.camera.center = (11, 12, 15)
    viewer.camera.zoom = 4

    np.testing.assert_almost_equal(viewer.camera.angles, (24, 12, -19))
    np.testing.assert_almost_equal(viewer.camera.center, (11, 12, 15))
    np.testing.assert_almost_equal(viewer.camera.zoom, 4)
    np.testing.assert_almost_equal(viewer.camera.angles, vispy_camera.angles)
    np.testing.assert_almost_equal(viewer.camera.center, vispy_camera.center)
    np.testing.assert_almost_equal(viewer.camera.zoom, vispy_camera.zoom)


def test_camera_model_update_from_vispy_3D(make_napari_viewer):
    """Test camera model updates from vispy in 3D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.vispy_camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    viewer.dims.ndisplay = 3

    # Update vispy camera angles, center, and zoom
    viewer.camera.angles = (24, 12, -19)
    vispy_camera.center = (11, 12, 15)
    vispy_camera.zoom = 4
    vispy_camera.on_draw(None)

    np.testing.assert_almost_equal(viewer.camera.angles, (24, 12, -19))
    np.testing.assert_almost_equal(viewer.camera.center, (11, 12, 15))
    np.testing.assert_almost_equal(viewer.camera.zoom, 4)
    np.testing.assert_almost_equal(viewer.camera.angles, vispy_camera.angles)
    np.testing.assert_almost_equal(viewer.camera.center, vispy_camera.center)
    np.testing.assert_almost_equal(viewer.camera.zoom, vispy_camera.zoom)

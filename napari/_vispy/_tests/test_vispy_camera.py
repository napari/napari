import numpy as np
from vispy.scene import ArcballCamera, PanZoomCamera
from vispy.util.quaternion import Quaternion


def test_camera(make_test_viewer):
    """Test vispy camera model interaction."""
    viewer = make_test_viewer()
    vispy_view = viewer.window.qt_viewer.view

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2
    assert viewer.camera.ndisplay == 2
    assert viewer.camera.center == (5.0, 5.0)
    assert viewer.camera.angles == (0, 0, 90)
    assert isinstance(vispy_view.camera, PanZoomCamera)
    assert vispy_view.camera.rect.center == (5.0, 5.0)
    assert vispy_view.camera.rect.size == (11, 11)

    # Change to 3D display and check vispy camera changes
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    assert viewer.camera.ndisplay == 3
    assert viewer.camera.center == (5.0, 5.0, 5.0)
    assert viewer.camera.angles == (0, 0, 90)
    assert isinstance(vispy_view.camera, ArcballCamera)
    assert vispy_view.camera.center == (5.0, 5.0, 5.0)
    assert vispy_view.camera.scale_factor == 11

    # Update camera model and check vispy camera changes in 3D
    viewer.camera.center = (20, 10, 15)
    viewer.camera.zoom = 2
    viewer.camera.angles = (-20, 10, -45)
    assert viewer.camera.ndisplay == 3
    assert viewer.camera.center == (20, 10, 15)
    assert viewer.camera.zoom == 2
    assert viewer.camera.angles == (-20, 10, -45)
    assert isinstance(vispy_view.camera, ArcballCamera)
    assert vispy_view.camera.center == (15, 10, 20)
    assert vispy_view.camera.scale_factor == 1200

    # Zoom and pan vispy camera and check camera model changes in 3D
    vispy_view.camera.center = (12, -2, 8)
    vispy_view.camera.scale_factor = 1800
    viewer.window.qt_viewer.on_draw(None)
    assert viewer.camera.center == (8, -2, 12)
    assert viewer.camera.zoom == 3

    # Update angle and check roundtrip is correct
    angles = (12, 53, 92)
    q = Quaternion.create_from_euler_angles(*angles, degrees=True)
    vispy_view.camera._quaternion = q
    viewer.window.qt_viewer.on_draw(None)
    np.testing.assert_allclose(viewer.camera.angles, angles)

    # Change back to 2D display and check vispy camera changes
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2
    assert viewer.camera.ndisplay == 2
    assert isinstance(vispy_view.camera, PanZoomCamera)

    # Update camera model and check vispy camera changes in 2D
    viewer.camera.center = (20, 30)
    viewer.camera.zoom = 4
    assert viewer.camera.ndisplay == 2
    assert viewer.camera.center == (20, 30)
    assert viewer.camera.zoom == 4
    assert isinstance(vispy_view.camera, PanZoomCamera)
    assert vispy_view.camera.rect.center == (30.0, 20.0)
    assert vispy_view.camera.rect.size == (2400.0, 2400.0)

    # Zoom and pan vispy camera and check camera model changes in 2D
    vispy_view.camera.zoom(2)
    viewer.window.qt_viewer.on_draw(None)
    assert vispy_view.camera.rect.size == (4800.0, 4800.0)
    assert viewer.camera.zoom == 8

    vispy_view.camera.zoom(0.5)
    viewer.window.qt_viewer.on_draw(None)
    assert vispy_view.camera.rect.size == (2400.0, 2400.0)
    assert viewer.camera.zoom == 4

    vispy_view.camera.rect = (-20, -30, 40, 10)
    viewer.window.qt_viewer.on_draw(None)
    assert viewer.camera.center == (-25, 0)
    assert viewer.camera.zoom == 0.05

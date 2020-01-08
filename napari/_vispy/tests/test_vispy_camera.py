import numpy as np
from napari import Viewer
from vispy.scene import PanZoomCamera, ArcballCamera


def test_camera(qtbot):
    """Test vispy camera model interaction."""
    viewer = Viewer()
    view = viewer.window.qt_viewer
    qtbot.addWidget(view)

    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2
    assert viewer.camera.ndisplay == 2
    assert viewer.camera.center == (5.0, 5.0)
    assert viewer.camera.angles == (0, 0, 90)
    assert isinstance(view.view.camera, PanZoomCamera)
    assert view.view.camera.rect.center == (5.0, 5.0)
    assert view.view.camera.rect.size == (11, 11)

    # Change to 3D display and check vispy camera changes
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    assert viewer.camera.ndisplay == 3
    assert viewer.camera.center == (5.0, 5.0, 5.0)
    assert viewer.camera.angles == (0, 0, 90)
    assert isinstance(view.view.camera, ArcballCamera)
    assert view.view.camera.center == (5.0, 5.0, 5.0)
    assert view.view.camera.scale_factor == 11

    # Update camera model and check vispy camera changes in 3D
    center = (20, 10, 15)
    scale = 100
    angles = (-20, 10, -45)
    viewer.camera.update(center=center, scale=scale, angles=angles)
    assert viewer.camera.ndisplay == 3
    assert viewer.camera.center == center
    assert viewer.camera.scale == scale
    assert viewer.camera.angles == angles
    assert isinstance(view.view.camera, ArcballCamera)
    assert view.view.camera.center == center[::-1]
    assert view.view.camera.scale_factor == 100

    # Zoom and pan vispy camera and check camera model changes in 2D
    view.view.camera.center = (12, -2, 8)
    view.view.camera.scale_factor = 20
    view.on_draw(None)
    assert viewer.camera.center == (8, -2, 12)
    assert viewer.camera.scale == 20

    # Change back to 2D display and check vispy camera changes
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2
    assert viewer.camera.ndisplay == 2
    assert isinstance(view.view.camera, PanZoomCamera)

    # Update camera model and check vispy camera changes in 2D
    center = (20, 30)
    scale = 200
    angles = (-20, 10, -45)
    viewer.camera.update(center=center, scale=scale, angles=angles)
    assert viewer.camera.ndisplay == 2
    assert viewer.camera.center == center
    assert viewer.camera.scale == scale
    assert viewer.camera.angles == angles
    assert isinstance(view.view.camera, PanZoomCamera)
    assert view.view.camera.rect.center == (30.0, 20.0)
    assert view.view.camera.rect.size == (200.0, 200.0)

    # Zoom and pan vispy camera and check camera model changes in 2D
    view.view.camera.zoom(2)
    view.on_draw(None)
    assert view.view.camera.rect.size == (400.0, 400.0)
    assert viewer.camera.scale == 400

    view.view.camera.zoom(0.5)
    view.on_draw(None)
    assert view.view.camera.rect.size == (200.0, 200.0)
    assert viewer.camera.scale == 200

    view.view.camera.rect = (-20, -30, 40, 10)
    view.on_draw(None)
    assert viewer.camera.center == (-25, 0)
    assert viewer.camera.scale == 40

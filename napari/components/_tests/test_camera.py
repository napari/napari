from napari.components import Camera


def test_camera():
    """Test camera."""
    camera = Camera()
    assert camera.center == (0, 0, 0)
    assert camera.zoom == 1
    assert camera.angles == (0, 0, 90)

    center = (10, 20, 30)
    camera.center = center
    assert camera.center == center
    assert camera.angles == (0, 0, 90)

    zoom = 200
    camera.zoom = zoom
    assert camera.zoom == zoom

    angles = (20, 90, 45)
    camera.angles = angles
    assert camera.angles == angles

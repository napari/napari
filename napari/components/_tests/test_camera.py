from napari.components import Camera, Dims


def test_camera():
    """Test camera."""
    camera = Camera(Dims(ndim=3))
    assert camera.ndisplay == 2
    assert camera.center == (0, 0)
    assert camera.zoom == 1
    assert camera.angles == (0, 0, 90)

    camera.ndisplay = 3
    center = (10, 20, 30)
    camera.center = center
    assert camera.ndisplay == 3
    assert camera.center == center
    assert camera.angles == (0, 0, 90)

    zoom = 200
    camera.zoom = zoom
    assert camera.zoom == zoom

    angles = (20, 90, 45)
    camera.angles = angles
    assert camera.angles == angles

import pytest

from napari.components import Camera


def test_camera():
    """Test camera."""
    camera = Camera()
    assert camera.ndisplay == 2
    assert camera.center == (0, 0)
    assert camera.size == 1
    assert camera.angles == (0, 0, 90)

    center = (10, 20, 30)
    camera.center = center
    assert camera.ndisplay == 3
    assert camera.center == center

    size = 200
    camera.size = size
    assert camera.size == size

    angles = (20, 90, 45)
    camera.angles = angles
    assert camera.angles == angles

    center = (20, 45)
    size = 300
    angles = (-20, 10, -45)
    camera.update(center=center, size=size, angles=angles)
    assert camera.ndisplay == 2
    assert camera.center == center
    assert camera.size == size
    assert camera.angles == angles

    with pytest.raises(ValueError):
        camera.center = (0,)

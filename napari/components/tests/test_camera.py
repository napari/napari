import pytest
from napari.components import Camera


def test_camera():
    """Test camera."""
    camera = Camera()
    assert camera.ndisplay == 2
    assert camera.center == (0, 0)
    assert camera.scale == 1
    assert camera.angles == (0, 0, 90)

    center = (10, 20, 30)
    camera.center = center
    assert camera.ndisplay == 3
    assert camera.center == center

    scale = 200
    camera.scale = scale
    assert camera.scale == scale

    angles = (20, 90, 45)
    camera.angles = angles
    assert camera.angles == angles

    center = (20, 45)
    scale = 300
    angles = (-20, 10, -45)
    camera.update(center=center, scale=scale, angles=angles)
    assert camera.ndisplay == 2
    assert camera.center == center
    assert camera.scale == scale
    assert camera.angles == angles

    with pytest.raises(ValueError):
        camera.center = (0,)

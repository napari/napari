import numpy as np

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


def test_calculate_view_direction_3d():
    """Check that view direction is calculated properly from camera angles."""
    # simple case
    camera = Camera(center=(0, 0, 0), angles=(90, 0, 0), zoom=1)
    assert np.allclose(camera.view_direction, (0, 1, 0))

    # shouldn't change with zoom
    camera = Camera(center=(0, 0, 0), angles=(90, 0, 0), zoom=10)
    assert np.allclose(camera.view_direction, (0, 1, 0))

    # shouldn't change with center
    camera = Camera(center=(15, 15, 15), angles=(90, 0, 0), zoom=1)
    assert np.allclose(camera.view_direction, (0, 1, 0))


def test_calculate_view_direction_nd():
    """Check that nD view direction is calculated properly."""
    camera = Camera(center=(0, 0, 0), angles=(90, 0, 0), zoom=1)

    # should return none if ndim == 2
    view_direction = camera.calculate_nd_view_direction(
        ndim=2, dims_displayed=[0, 1]
    )
    assert view_direction is None

    # should return 3d if ndim == 3
    view_direction = camera.calculate_nd_view_direction(
        ndim=3, dims_displayed=[0, 1, 2]
    )
    assert len(view_direction) == 3
    assert np.allclose(view_direction, (0, 1, 0))

    # should return nD with 3d embedded in nD if ndim > 3
    view_direction = camera.calculate_nd_view_direction(
        ndim=5, dims_displayed=[0, 2, 4]
    )
    assert len(view_direction) == 5
    assert np.allclose(view_direction[[0, 2, 4]], (0, 1, 0))

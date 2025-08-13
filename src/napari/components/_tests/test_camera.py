from itertools import product

import numpy as np
import pytest

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


def test_calculate_up_direction_3d():
    """Check that up direction is calculated properly from camera angles."""
    # simple case
    camera = Camera(center=(0, 0, 0), angles=(0, 0, 90), zoom=1)
    assert np.allclose(camera.up_direction, (0, -1, 0))

    # shouldn't change with zoom
    camera = Camera(center=(0, 0, 0), angles=(0, 0, 90), zoom=10)
    assert np.allclose(camera.up_direction, (0, -1, 0))

    # shouldn't change with center
    camera = Camera(center=(15, 15, 15), angles=(0, 0, 90), zoom=1)
    assert np.allclose(camera.up_direction, (0, -1, 0))

    # more complex case with order dependent Euler angles
    camera = Camera(center=(0, 0, 0), angles=(10, 20, 30), zoom=1)
    assert np.allclose(camera.up_direction, (-0.88, -0.44, 0.16), atol=0.01)


def _normalize_angle(degrees):
    """Normalize angle to be in (-180, 180]."""
    return -((180 - np.asarray(degrees)) % 360) + 180


def test_set_view_direction_3d():
    """Check that view direction can be set properly."""
    # simple case
    camera = Camera(center=(0, 0, 0), angles=(0, 0, 0), zoom=1)
    camera.set_view_direction(view_direction=(1, 0, 0))
    assert np.allclose(camera.view_direction, (1, 0, 0))
    assert np.allclose(_normalize_angle(camera.angles), (180, 0, -90))

    # case with ordering and up direction setting
    view_direction = np.array([1, 2, 3], dtype=float)
    view_direction /= np.linalg.norm(view_direction)
    camera.set_view_direction(view_direction=view_direction)
    assert np.allclose(camera.view_direction, view_direction)
    assert np.allclose(
        _normalize_angle(camera.angles), (121.9, -53.3, -26.6), atol=0.1
    )


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


@pytest.mark.parametrize(
    ('orientation', 'expected_handedness'),
    zip(
        # Could do this but the order is not locally visible, so we explicitly
        # order the strings so we can think about handedness locally.
        # product(DepthAxisOrientation, VerticalAxisOrientation, HorizontalAxisOrientation),
        product(['towards', 'away'], ['down', 'up'], ['right', 'left']),
        # hardcoded after running once
        ['right', 'left', 'left', 'right', 'left', 'right', 'right', 'left'],
        strict=False,
    ),
)
def test_handedness(orientation, expected_handedness):
    """Check that handedness is calculated properly."""
    camera = Camera()
    camera.orientation = orientation
    assert camera.handedness == expected_handedness

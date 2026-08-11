"""Tests for the shared camera orientation math in ``napari.utils.camera_orientations``."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from napari.utils.camera_orientations import (
    angles_from_view_direction,
    up_direction_from_angles,
    view_direction_from_angles,
)

ORIENTATIONS = [
    (depth, vertical, horizontal)
    for depth in ['towards', 'away']
    for vertical in ['down', 'up']
    for horizontal in ['right', 'left']
]

ANGLES = st.tuples(
    *[
        st.floats(
            min_value=-180,
            max_value=180,
            allow_nan=False,
            allow_infinity=False,
        )
        for _ in range(3)
    ]
)


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_home_view_matches_base_directions(orientation):
    """With zero angles the camera must show the home view."""
    depth, vertical, _ = orientation
    assert np.allclose(
        view_direction_from_angles((0, 0, 0), orientation),
        (-1 if str(depth) == 'towards' else 1, 0, 0),
    )
    assert np.allclose(
        up_direction_from_angles((0, 0, 0), orientation),
        (0, -1 if str(vertical) == 'down' else 1, 0),
    )


@pytest.mark.parametrize(
    ('angle_index', 'axis_component'),
    [
        # rx rotates about the depth axis (dim0)
        (0, 0),
        # ry rotates about the vertical axis (dim1)
        (1, 1),
        # rz rotates about the horizontal axis (dim2)
        (2, 2),
    ],
)
def test_single_angle_rotation_axis(angle_index, axis_component):
    """A single nonzero angle must rotate the camera about the expected axis."""
    orientation = ('towards', 'down', 'right')

    def camera_basis(angles):
        view = np.asarray(view_direction_from_angles(angles, orientation))
        up = np.asarray(up_direction_from_angles(angles, orientation))
        return np.stack([view, up, np.cross(view, up)], axis=1)

    base_basis = camera_basis((0, 0, 0))
    angles = [0, 0, 0]
    angles[angle_index] = 20
    rotation = camera_basis(tuple(angles)) @ base_basis.T
    from scipy.spatial.transform import Rotation

    axis = Rotation.from_matrix(rotation).as_rotvec()
    axis /= np.linalg.norm(axis)
    assert np.allclose(np.abs(axis), np.eye(3)[axis_component])


@pytest.mark.parametrize('orientation', ORIENTATIONS)
@given(angles=ANGLES)
@settings(max_examples=50, deadline=None)
def test_angles_from_view_direction_roundtrip(orientation, angles):
    """Check that angles_from_view_direction inverts view/up direction."""
    view_direction = view_direction_from_angles(angles, orientation)
    up_direction = up_direction_from_angles(angles, orientation)
    recovered = angles_from_view_direction(
        view_direction, up_direction, orientation
    )
    # Euler angles are degenerate, so compare the resulting directions
    # rather than the angles themselves.
    assert np.allclose(
        view_direction_from_angles(recovered, orientation), view_direction
    )
    assert np.allclose(
        up_direction_from_angles(recovered, orientation), up_direction
    )


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_angles_from_view_direction_non_orthogonal_up(orientation):
    """Check that a non-orthogonal up direction is projected correctly."""
    view_direction = (1.0, 0.5, 0.25)
    up_direction = (0.0, 1.0, 0.0)
    angles = angles_from_view_direction(
        view_direction, up_direction, orientation
    )
    # the recovered up direction should be orthogonal to the view direction
    # and point "up" on the canvas
    up = up_direction_from_angles(angles, orientation)
    assert np.allclose(
        np.dot(up, view_direction_from_angles(angles, orientation)), 0
    )
    assert np.dot(up, up_direction) > 0

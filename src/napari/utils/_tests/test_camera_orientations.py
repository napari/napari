"""Tests for the shared camera orientation math in ``napari.utils.camera_orientations``."""

import os

import numpy as np
import pytest

from napari.utils.camera_orientations import (
    _get_vispy_flipped_axes,
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


@pytest.mark.filterwarnings('ignore:gimbal lock')
@pytest.mark.skipif(
    'GITHUB_ACTIONS' in os.environ, reason='Too slow for GitHub Actions'
)
def test_view_direction_correct_under_rotation():
    """Check that the shared math matches a real VisPy 3D camera.

    The orientation math in ``napari.utils.camera_orientations`` is a pure
    re-implementation of the transformations applied by the VisPy 3D camera
    (quaternion rotation and axis flips). This test verifies that the two
    agree for random angles across all orientations.
    """
    from vispy import scene

    from napari._vispy.camera import napari_angles_to_vispy_quat

    rng = np.random.default_rng()
    canvas = scene.SceneCanvas(size=(100, 100), show=False)
    try:
        view = canvas.central_widget.add_view()
        camera = scene.ArcballCamera(fov=0)
        view.camera = camera
        camera.set_range(x=(0, 10), y=(0, 10), z=(0, 10))
        for orientation in ORIENTATIONS:
            flips = _get_vispy_flipped_axes(orientation, ndisplay=3)
            camera.flip = flips
            for _ in range(25):
                angles = tuple(90 * rng.random(3))
                camera.set_state(
                    _quaternion=napari_angles_to_vispy_quat(angles, flips)
                )
                camera.view_changed()
                matrix_inv = np.linalg.inv(camera.transform.matrix[:3, :3])
                # VisPy uses xyz coordinates; napari uses zyx, hence the reversal.
                view_direction = (-matrix_inv[:, 2])[::-1]
                up_direction = (matrix_inv[:, 1])[::-1]
                assert np.allclose(
                    view_direction_from_angles(angles, flips), view_direction
                )
                assert np.allclose(
                    up_direction_from_angles(angles, flips), up_direction
                )
    finally:
        canvas.close()


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_angles_from_view_direction_roundtrip(orientation):
    """Check that angles_from_view_direction inverts view/up direction."""
    flips = _get_vispy_flipped_axes(orientation, ndisplay=3)
    rng = np.random.default_rng()
    for _ in range(20):
        angles = tuple(90 * rng.random(3))
        view_direction = view_direction_from_angles(angles, flips)
        up_direction = up_direction_from_angles(angles, flips)
        recovered = angles_from_view_direction(
            view_direction, up_direction, flips
        )
        # Euler angles are degenerate, so compare the resulting directions
        # rather than the angles themselves.
        assert np.allclose(
            view_direction_from_angles(recovered, flips), view_direction
        )
        assert np.allclose(
            up_direction_from_angles(recovered, flips), up_direction
        )


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_angles_from_view_direction_non_orthogonal_up(orientation):
    """Check that a non-orthogonal up direction is projected correctly."""
    flips = _get_vispy_flipped_axes(orientation, ndisplay=3)
    view_direction = (1.0, 0.5, 0.25)
    up_direction = (0.0, 1.0, 0.0)
    angles = angles_from_view_direction(view_direction, up_direction, flips)
    # the recovered up direction should be orthogonal to the view direction
    # and point "up" on the canvas
    up = up_direction_from_angles(angles, flips)
    assert np.allclose(
        np.dot(up, view_direction_from_angles(angles, flips)), 0
    )
    assert np.dot(up, up_direction) > 0

import numpy as np
from vispy.util.quaternion import Quaternion

from napari._vispy.quaternion import quaternion2euler


def test_quaternion2euler():
    """Test quaternion to euler angle conversion."""
    # Test some sets of angles
    angles_test = [[12, 53, 92], [180, -90, 0]]

    for angles in angles_test:
        q = Quaternion.create_from_euler_angles(*angles, degrees=True)
        ea = quaternion2euler(q, degrees=True)
        q_p = Quaternion.create_from_euler_angles(*ea, degrees=True)
        np.testing.assert_allclose(
            list(q.__dict__.values()), list(q_p.__dict__.values())
        )

        # Test roundtrip radians
        angles_rad = np.radians(angles)
        q = Quaternion.create_from_euler_angles(*angles_rad)
        ea = quaternion2euler(q)
        q_p = Quaternion.create_from_euler_angles(*ea)
        np.testing.assert_allclose(
            list(q.__dict__.values()), list(q_p.__dict__.values())
        )

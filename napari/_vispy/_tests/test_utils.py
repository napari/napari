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
        np.testing.assert_allclose(ea, angles)

        # Test roundtrip radians
        angles_rad = np.radians(angles)
        q = Quaternion.create_from_euler_angles(*angles_rad)
        ea = quaternion2euler(q)
        np.testing.assert_allclose(ea, angles_rad)

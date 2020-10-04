import numpy as np
from vispy.util.quaternion import Quaternion

from napari._vispy.quaternion import quaternion2euler


def test_quaternion2euler(qtbot):
    """Test quaternion to euler angle conversion."""
    # Test roundtrip degrees
    angles = (12, 53, 92)
    q = Quaternion.create_from_euler_angles(*angles, degrees=True)
    ea = quaternion2euler(q, degrees=True)
    np.testing.assert_allclose(ea, angles)

    # Test roundtrip radians
    angles = (0.1, -0.2, 1.2)
    q = Quaternion.create_from_euler_angles(*angles)
    ea = quaternion2euler(q)
    np.testing.assert_allclose(ea, angles)

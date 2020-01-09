from napari._vispy import quaternion2euler
from vispy.util.quaternion import Quaternion
import numpy as np


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

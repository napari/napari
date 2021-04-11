import numpy as np
import pytest
from vispy.util.quaternion import Quaternion

from napari._vispy.quaternion import quaternion2euler


@pytest.mark.parametrize('angles', [[12, 53, 92], [180, -90, 0], [16, 90, 0]])
def test_quaternion2euler(angles):
    """Test quaternion to euler angle conversion."""

    # Test for degrees
    q = Quaternion.create_from_euler_angles(*angles, degrees=True)
    ea = quaternion2euler(q, degrees=True)
    q_p = Quaternion.create_from_euler_angles(*ea, degrees=True)

    # We now compare the corresponding quaternions ; they should be equals or opposites (as they're already unit ones)
    q_values = np.array([q.w, q.x, q.y, q.z])
    q_p_values = np.array([q_p.w, q_p.x, q_p.y, q_p.z])

    nn_zero_ind = np.argmax((q_values != 0) & (q_p_values != 0))

    q_values *= np.sign(q_values[nn_zero_ind])
    q_p_values *= np.sign(q_p_values[nn_zero_ind])

    np.testing.assert_allclose(q_values, q_p_values)

    # Test for radians
    angles_rad = np.radians(angles)
    q = Quaternion.create_from_euler_angles(*angles_rad)
    ea = quaternion2euler(q)
    q_p = Quaternion.create_from_euler_angles(*ea)

    # We now compare the corresponding quaternions ; they should be equals or opposites (as they're already unit ones)
    q_values = np.array([q.w, q.x, q.y, q.z])
    q_p_values = np.array([q_p.w, q_p.x, q_p.y, q_p.z])

    nn_zero_ind = np.argmax((q_values != 0) & (q_p_values != 0))

    q_values *= np.sign(q_values[nn_zero_ind])
    q_p_values *= np.sign(q_p_values[nn_zero_ind])

    np.testing.assert_allclose(q_values, q_p_values)

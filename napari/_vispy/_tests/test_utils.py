import numpy as np
import pytest
from vispy.util.quaternion import Quaternion

from napari._vispy.quaternion import quaternion2euler

# Euler angles to be tested, in degrees
angles = [[12, 53, 92], [180, -90, 0], [16, 90, 0]]

# Prepare for input and add corresponding values in radians
angles_param = [(x, True) for x in angles]
angles_param.extend([(x, False) for x in np.radians(angles)])


@pytest.mark.parametrize('angles,degrees', angles_param)
def test_quaternion2euler(angles, degrees):
    """Test quaternion to euler angle conversion."""

    # Test for degrees
    q = Quaternion.create_from_euler_angles(*angles, degrees)
    ea = quaternion2euler(q, degrees=degrees)
    q_p = Quaternion.create_from_euler_angles(*ea, degrees=degrees)

    # We now compare the corresponding quaternions ; they should be equals or opposites (as they're already unit ones)
    q_values = np.array([q.w, q.x, q.y, q.z])
    q_p_values = np.array([q_p.w, q_p.x, q_p.y, q_p.z])

    nn_zero_ind = np.argmax((q_values != 0) & (q_p_values != 0))

    q_values *= np.sign(q_values[nn_zero_ind])
    q_p_values *= np.sign(q_p_values[nn_zero_ind])

    np.testing.assert_allclose(q_values, q_p_values)

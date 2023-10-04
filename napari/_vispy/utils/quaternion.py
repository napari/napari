import warnings

from scipy.spatial.transform import Rotation


def quaternion2euler_degrees(quaternion):
    """Converts VisPy quaternion into euler angle representation.

    Euler angles have degeneracies, so the output might different
    from the Euler angles that might have been used to generate
    the input quaternion.


    Parameters
    ----------
    quaternion : vispy.util.Quaternion
        Quaternion for conversion.

    Returns
    -------
    angles : 3-tuple
        Euler angles in (rx, ry, rz) order.
    """
    q = quaternion
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='Gimbal lock detected.*',
            category=UserWarning,
        )

        # despite both SciPy and  previous implementation advertising
        # the  x,y,z order, it appear we need to reverse the order to get
        # the same results a previously.
        # as_euler('zyx') does not work as composition order matters
        z, y, x = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler(
            'xyz', degrees=True
        )
    return x, y, z

import numpy as np
import scipy.linalg

from ...utils.translations import trans


def coerce_rotate(rotate):
    if np.isscalar(rotate):
        return _make_2d_rotation(rotate)
    elif np.array(rotate).ndim == 1 and len(rotate) == 3:
        return _make_3d_rotation(*rotate)
    return np.array(rotate)


def _make_2d_rotation(theta_degrees):
    """Makes a 2D rotation matrix from an angle in degrees."""
    theta = np.deg2rad(theta_degrees)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def _make_3d_rotation(alpha_deg, beta_deg, gamma_deg):
    """Makes a 3D rotation matrix from roll, pitch, and yaw in degrees.

    For more details, see: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    """
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)
    R_alpha = np.array(
        [
            [np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    R_beta = np.array(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )
    R_gamma = np.array(
        [
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)],
        ]
    )
    return R_alpha @ R_beta @ R_gamma


def coerce_translate(ndim, translate):
    translate_arr = np.zeros(ndim)
    if translate is not None:
        translate_arr[ndim - len(translate) :] = translate
    return translate_arr


def coerce_scale(ndim, scale):
    scale_arr = np.ones(ndim)
    if scale is not None:
        scale_arr[ndim - len(scale) :] = scale
    return scale_arr


def coerce_shear(shear):
    # Check if an upper-triangular representation of shear or
    # a full nD shear matrix has been passed
    if np.isscalar(shear):
        raise ValueError(
            trans._(
                'Scalars are not valid values for shear. Shear must be an upper triangular vector or square matrix with ones along the main diagonal.',
                deferred=True,
            )
        )
    if np.array(shear).ndim == 1:
        return expand_upper_triangular(shear)

    if not is_matrix_triangular(shear):
        raise ValueError(
            trans._(
                'Only upper triangular or lower triangular matrices are accepted for shear, got {shear}. For other matrices, set the affine_matrix or linear_matrix directly.',
                deferred=True,
                shear=shear,
            )
        )
    return np.array(shear)


def expand_upper_triangular(vector):
    """Expand a vector into an upper triangular matrix.

    Decomposition is based on code from https://github.com/matthew-brett/transforms3d.
    In particular, the `striu2mat` function in the `shears` module.
    https://github.com/matthew-brett/transforms3d/blob/0.3.1/transforms3d/shears.py#L30-L77.

    Parameters
    ----------
    vector : np.array
        1D vector of length M

    Returns
    -------
    upper_tri : np.array shape (N, N)
        Upper triangular matrix.
    """
    n = len(vector)
    N = ((-1 + np.sqrt(8 * n + 1)) / 2.0) + 1  # n+1 th root
    if N != np.floor(N):
        raise ValueError(
            trans._(
                '{number} is a strange number of shear elements',
                deferred=True,
                number=n,
            )
        )

    N = int(N)
    inds = np.triu(np.ones((N, N)), 1).astype(bool)
    upper_tri = np.eye(N)
    upper_tri[inds] = vector
    return upper_tri


def shear_matrix_from_angle(angle, ndim=3, axes=(-1, 0)):
    """Create a shear matrix from an angle.

    Parameters
    ----------
    angle : float
        Angle in degrees.
    ndim : int
        Dimensionality of the shear matrix
    axes : 2-tuple of int
        Location of the angle in the shear matrix.
        Default is the lower left value.

    Returns
    -------
    matrix : np.ndarray
        Shear matrix with ones along the main diagonal
    """
    matrix = np.eye(ndim)
    matrix[axes] = np.tan(np.deg2rad(90 - angle))
    return matrix


def is_matrix_upper_triangular(matrix):
    """Check if a matrix is upper triangular.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Whether matrix is upper triangular or not.
    """
    return np.allclose(matrix, np.triu(matrix))


def is_matrix_lower_triangular(matrix):
    """Check if a matrix is lower triangular.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Whether matrix is lower triangular or not.
    """
    return np.allclose(matrix, np.tril(matrix))


def is_matrix_triangular(matrix):
    """Check if a matrix is triangular.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be checked.

    Returns
    -------
    bool
        Whether matrix is triangular or not.
    """
    return is_matrix_upper_triangular(matrix) or is_matrix_lower_triangular(
        matrix
    )


def decompose_linear_matrix(
    matrix, upper_triangular=True
) -> (np.array, np.array, np.array):
    """Decompose linear transform matrix into rotate, scale, shear.
    Decomposition is based on code from https://github.com/matthew-brett/transforms3d.
    In particular, the `decompose` function in the `affines` module.
    https://github.com/matthew-brett/transforms3d/blob/0.3.1/transforms3d/affines.py#L156-L246.
    Parameters
    ----------
    matrix : np.array shape (N, N)
        nD array representing the composed linear transform.
    upper_triangular : bool
        Whether to decompose shear into an upper triangular or
        lower triangular matrix.
    Returns
    -------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : 1-D array or n-D array
        1-D array of upper triangular values or an n-D matrix if lower
        triangular.
    """
    n = matrix.shape[0]

    if upper_triangular:
        rotate, tri = scipy.linalg.qr(matrix)
    else:
        upper_tri, rotate = scipy.linalg.rq(matrix.T)
        rotate = rotate.T
        tri = upper_tri.T

    scale = np.diag(tri).copy()

    # Take any reflection into account
    if np.linalg.det(rotate) < 0:
        scale[0] *= -1
        tri[0] *= -1
        rotate = matrix @ np.linalg.inv(tri)

    tri_normalized = tri @ np.linalg.inv(np.diag(scale))

    if upper_triangular:
        shear = tri_normalized[np.triu(np.ones((n, n)), 1).astype(bool)]
    else:
        shear = tri_normalized

    return rotate, scale, shear

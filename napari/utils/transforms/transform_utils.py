import numpy as np
import scipy.linalg

from ...utils.translations import trans


def compose_linear_matrix(rotate, scale, shear) -> np.array:
    """Compose linear transform matrix from rotate, shear, scale.

    Parameters
    ----------
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
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.

    Returns
    -------
    matrix : array
        nD array representing the composed linear transform.
    """
    if np.isscalar(rotate):
        # If a scalar is passed assume it is a single rotation angle
        # for a 2D rotation
        theta = np.deg2rad(rotate)
        rotate_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    elif np.array(rotate).ndim == 1 and len(rotate) == 3:
        # If a 3-tuple is passed assume it is three rotation angles for
        # a roll, pitch, and yaw for a 3D rotation. For more details see
        # https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
        alpha = np.deg2rad(rotate[0])
        beta = np.deg2rad(rotate[1])
        gamma = np.deg2rad(rotate[2])
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
        rotate_mat = R_alpha @ R_beta @ R_gamma
    else:
        # Otherwise assume a full nD rotation matrix has been passed
        rotate_mat = np.array(rotate)
    n_rotate = rotate_mat.shape[0]

    # Convert a scale vector to an nD diagonal matrix
    scale_mat = np.diag(scale)
    n_scale = scale_mat.shape[0]

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
        shear_mat = expand_upper_triangular(shear)
    else:
        shear_mat = np.array(shear)

    # Check the dimensionality of the transforms and pad as needed
    n_shear = shear_mat.shape[0]
    ndim = max(n_scale, n_rotate, n_shear)

    full_scale = embed_in_identity_matrix(scale_mat, ndim)
    full_rotate = embed_in_identity_matrix(rotate_mat, ndim)
    full_shear = embed_in_identity_matrix(shear_mat, ndim)
    return full_rotate @ full_shear @ full_scale


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
        Upper triangluar matrix.
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


def embed_in_identity_matrix(matrix, ndim):
    """Embed an MxM matrix bottom right of larger NxN identity matrix.

    Parameters
    ----------
    matrix : np.array
        2D square matrix, MxM.
    ndim : int
        Integer with N >= M.

    Returns
    -------
    full_matrix : np.array shape (N, N)
        Larger matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            trans._(
                'Improper transform matrix {matrix}',
                deferred=True,
                matrix=matrix,
            )
        )

    if matrix.shape[0] == ndim:
        return matrix
    else:
        full_matrix = np.eye(ndim)
        full_matrix[-matrix.shape[0] :, -matrix.shape[1] :] = matrix
        return full_matrix


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

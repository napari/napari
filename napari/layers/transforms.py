from typing import Sequence

import numpy as np
import toolz as tz

from ..utils.list import ListModel


class Transform:
    """Base transform class.

    Defaults to the identity transform.

    Parameters
    ----------
    func : callable, Coords -> Coords
        A function converting an NxD array of coordinates to NxD'.
    name : string
        A string name for the transform.
    """

    def __init__(self, func=tz.identity, inverse=None, name=None):
        self.func = func
        self._inverse_func = inverse
        self.name = name

        if func is tz.identity:
            self._inverse_func = tz.identity

    def __call__(self, coords):
        """Transform input coordinates to output."""
        return self.func(coords)

    @property
    def inverse(self) -> 'Transform':
        if self._inverse_func is not None:
            return Transform(self._inverse_func, self.func)
        else:
            raise ValueError('Inverse function was not provided.')

    def compose(self, transform: 'Transform') -> 'Transform':
        """Return the composite of this transform and the provided one."""
        raise ValueError('Transform composition rule not provided')

    def set_slice(self, axes: Sequence[int]) -> 'Transform':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Transform
            Resulting transform.
        """
        raise NotImplementedError('Cannot subset arbitrary transforms.')

    def expand_dims(self, axes: Sequence[int]) -> 'Transform':
        """Return a transform with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        Transform
            Resulting transform.
        """
        raise NotImplementedError('Cannot subset arbitrary transforms.')


class TransformChain(ListModel, Transform):
    def __init__(self, transforms=[]):
        super().__init__(
            basetype=Transform,
            iterable=transforms,
            lookup={str: lambda q, e: q == e.name},
        )

    def __call__(self, coords):
        return tz.pipe(coords, *self)

    def __newlike__(self, iterable):
        return ListModel(self._basetype, iterable, self._lookup)

    @property
    def inverse(self) -> 'TransformChain':
        """Return the inverse transform chain."""
        return TransformChain([tf.inverse for tf in self[::-1]])

    @property
    def simplified(self) -> 'Transform':
        """Return the composite of the transforms inside the transform chain."""
        if len(self) == 0:
            return None
        if len(self) == 1:
            return self[0]
        else:
            return tz.pipe(self[0], *[tf.compose for tf in self[1:]])

    def set_slice(self, axes: Sequence[int]) -> 'TransformChain':
        """Return a transform chain subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform chain with.

        Returns
        -------
        TransformChain
            Resulting transform chain.
        """
        return TransformChain([tf.set_slice(axes) for tf in self])

    def expand_dims(self, axes: Sequence[int]) -> 'Transform':
        """Return a transform chain with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        TransformChain
            Resulting transform chain.
        """
        return TransformChain([tf.expand_dims(axes) for tf in self])


class ScaleTranslate(Transform):
    """n-dimensional scale and translation (shift) class.

    Scaling is always applied before translation.

    Parameters
    ----------
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    name : string
        A string name for the transform.
    """

    def __init__(self, scale=(1.0,), translate=(0.0,), *, name=None):
        super().__init__(name=name)
        self.scale = np.array(scale)
        self.translate = np.array(translate)

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        scale = np.concatenate(
            ([1.0] * (coords.shape[1] - len(self.scale)), self.scale)
        )
        translate = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.translate)), self.translate)
        )
        return np.atleast_1d(np.squeeze(scale * coords + translate))

    @property
    def inverse(self) -> 'ScaleTranslate':
        """Return the inverse transform."""
        return ScaleTranslate(1 / self.scale, -1 / self.scale * self.translate)

    def compose(self, transform: 'ScaleTranslate') -> 'ScaleTranslate':
        """Return the composite of this transform and the provided one."""
        scale = self.scale * transform.scale
        translate = self.translate + self.scale * transform.translate
        return ScaleTranslate(scale, translate)

    def set_slice(self, axes: Sequence[int]) -> 'ScaleTranslate':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Transform
            Resulting transform.
        """
        return ScaleTranslate(
            self.scale[axes], self.translate[axes], name=self.name
        )

    def expand_dims(self, axes: Sequence[int]) -> 'ScaleTranslate':
        """Return a transform with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        Transform
            Resulting transform.
        """
        n = len(axes) + len(self.scale)
        not_axes = [i for i in range(n) if i not in axes]
        scale = np.ones(n)
        scale[not_axes] = self.scale
        translate = np.zeros(n)
        translate[not_axes] = self.translate
        return ScaleTranslate(scale, translate, name=self.name)


class Affine(Transform):
    """n-dimensional affine transformation class.

    The affine transform is represented as a n+1 dimensionsal matrix

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotate matrix using that value as an
        angle. If 3-tuple convert into a 3D rotate matrix, rolling a yaw,
        pitch, roll convention. Otherwise assume an nD rotate. Angle
        conversion are done either using degrees or radians depending on the
        degrees boolean parameter.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : n-D array
        An n-D shear matrix.
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    degrees : bool
        Boolean if rotate angles are provided in degrees
    name : string
        A string name for the transform.
    """

    def __init__(
        self,
        scale=(1.0,),
        translate=(0.0,),
        *,
        rotate=None,
        shear=None,
        matrix=None,
        degrees=True,
        name=None,
    ):
        super().__init__(name=name)

        if matrix is None:
            if rotate is None:
                rotate = np.eye(len(scale))
            if shear is None:
                shear = np.eye(len(scale))
            matrix = compose_linear_matrix(
                rotate, scale, shear, degrees=degrees
            )
        else:
            matrix = np.array(matrix)

        ndim = max(matrix.shape[0], len(translate))
        self.matrix = embed_in_identity_matrix(matrix, ndim)
        self.translate = np.array(
            [0] * (ndim - len(translate)) + list(translate)
        )

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        if coords.shape[1] != self.matrix.shape[0]:
            matrix = np.eye(coords.shape[1])
            matrix[
                -self.matrix.shape[0] :, -self.matrix.shape[1] :
            ] = self.matrix
        else:
            matrix = self.matrix
        translate = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.translate)), self.translate)
        )
        return np.atleast_1d(np.squeeze(coords @ matrix + translate))

    @property
    def scale(self) -> np.array:
        """Return the scale of the transform."""
        return decompose_linear_matrix(self.matrix)[1]

    @scale.setter
    def scale(self, scale):
        """Set the scale of the transform."""
        rotate, _, shear = decompose_linear_matrix(self.matrix)
        self.matrix = compose_linear_matrix(rotate, scale, shear)

    @property
    def rotate(self) -> np.array:
        """Return the rotate of the transform."""
        return decompose_linear_matrix(self.matrix)[0]

    @rotate.setter
    def rotate(self, rotate):
        """Set the rotate of the transform."""
        _, scale, shear = decompose_linear_matrix(self.matrix)
        self.matrix = compose_linear_matrix(rotate, scale, shear)

    @property
    def shear(self) -> np.array:
        """Return the shear of the transform."""
        return decompose_linear_matrix(self.matrix)[2]

    @shear.setter
    def shear(self, shear):
        """Set the rotate of the transform."""
        rotate, scale, _ = decompose_linear_matrix(self.matrix)
        self.matrix = compose_linear_matrix(rotate, scale, shear)

    @property
    def inverse(self) -> 'Affine':
        """Return the inverse transform."""
        matrix = np.linalg.inv(self.matrix)
        translate = -self.translate @ matrix
        return Affine(matrix=matrix, translate=translate)

    def compose(self, transform: 'Affine') -> 'Affine':
        """Return the composite of this transform and the provided one."""
        matrix = transform.matrix @ self.matrix
        translate = self.translate + transform.translate @ self.matrix
        return Affine(matrix=matrix, translate=translate)

    def set_slice(self, axes: Sequence[int]) -> 'Affine':
        """Return a transform subset to the visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Axes to subset the current transform with.

        Returns
        -------
        Transform
            Resulting transform.
        """
        return Affine(
            matrix=self.matrix[np.ix_(axes, axes)],
            translate=self.translate[axes],
            name=self.name,
        )

    def expand_dims(self, axes: Sequence[int]) -> 'Affine':
        """Return a transform with added axes for non-visible dimensions.

        Parameters
        ----------
        axes : Sequence[int]
            Location of axes to expand the current transform with. Passing a
            list allows expansion to occur at specific locations and for
            expand_dims to be like an inverse to the set_slice method.

        Returns
        -------
        Transform
            Resulting transform.
        """
        n = len(axes) + len(self.scale)
        not_axes = [i for i in range(n) if i not in axes]
        matrix = np.eye(n)
        matrix[np.ix_(not_axes, not_axes)] = self.matrix
        translate = np.zeros(n)
        translate[not_axes] = self.translate
        return Affine(matrix=matrix, translate=translate, name=self.name)


def compose_linear_matrix(rotate, scale, shear, degrees=True) -> np.array:
    """Compose linear transform matrix from rotate, shear, scale.

    Parameters
    ----------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotate matrix using that value as an
        angle. If 3-tuple convert into a 3D rotate matrix, rolling a yaw,
        pitch, roll convention. Otherwise assume an nD rotate. Angle
        conversion are done either using degrees or radians depending on the
        degrees boolean parameter.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : n-D array
        An n-D shear matrix.

    Returns
    -------
    matrix : array
        nD array representing the composed linear transform.
    """
    if np.isscalar(rotate):
        # If a scalar is passed assume it is a single rotate angle
        # for a 2D rotate
        if degrees:
            theta = np.deg2rad(rotate)
        else:
            theta = rotate
        rotate_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    elif np.array(rotate).ndim == 1 and len(rotate) == 3:
        # If a 3-tuple is passed assume it is three rotation angles for
        # a roll, pitch, and yaw for a 3D rotation. For more details see
        # https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
        if degrees:
            alpha = np.deg2rad(rotate[0])
            beta = np.deg2rad(rotate[1])
            gamma = np.deg2rad(rotate[2])
        else:
            alpha = rotate[0]
            beta = rotate[1]
            gamma = rotate[2]
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
        # Otherwise assume a full nD rotate matrix has been passed
        rotate_mat = np.array(rotate)

    # Convert a scale vector to an nD diagonal matrix
    scale_mat = np.diag(scale)

    # Check if an upper-triangular representation of shear or
    # a full nD shear matrix has been passed
    if np.array(shear).ndim == 1:
        shear_mat = expand_upper_triangular(shear)
    else:
        shear_mat = np.array(shear)

    # Check the dimensionality of the transforms and pad as needed
    n_scale = scale_mat.shape[0]
    n_rotate = rotate_mat.shape[0]
    n_shear = shear_mat.shape[0]
    ndim = max(n_scale, n_rotate, n_shear)

    full_scale = embed_in_identity_matrix(scale_mat, ndim)
    full_rotate = embed_in_identity_matrix(rotate_mat, ndim)
    full_shear = embed_in_identity_matrix(shear_mat, ndim)
    return full_shear @ full_scale @ full_rotate


def expand_upper_triangular(vector):
    """Expand a vector into an upper triangular matrix.

    Decomposition is modeled on code from https://github.com/matthew-brett/transforms3d.
    In particular, https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/shears.py#L30.

    Parameters
    ----------
    vector : np.array
        1D vector of length M

    Returns
    -------
    matrix : np.array shape (N, N)
        Upper triangluar matrix.
    """
    n = len(vector)
    N = ((-1 + np.sqrt(8 * n + 1)) / 2.0) + 1  # n+1 th root
    if N != np.floor(N):
        raise ValueError('%d is a strange number of shear elements' % n)
    N = int(N)
    inds = np.triu(np.ones((N, N)), 1).astype(bool)
    matrix = np.eye(N)
    matrix[inds] = vector
    return matrix


def embed_in_identity_matrix(matrix, ndim):
    """Embed an MxM matrix in a larger NxN identity matrix.

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
        raise ValueError(f'Improper transform matrix {matrix}')

    if matrix.shape[0] == ndim:
        return matrix
    else:
        full_matrix = np.eye(ndim)
        full_matrix[-matrix.shape[0] :, -matrix.shape[1] :] = matrix
        return full_matrix


def decompose_linear_matrix(matrix) -> (np.array, np.array, np.array):
    """Decompose linear transform matrix into rotate, scale, shear.

    Decomposition is modeled on code from https://github.com/matthew-brett/transforms3d.
    In particular, https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/affines.py#L156.

    Parameters
    ----------
    matrix : np.array shape (N, N)
        nD array representing the composed linear transform.

    Returns
    -------
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotate matrix using that value as an
        angle. If 3-tuple convert into a 3D rotate matrix, rolling a yaw,
        pitch, roll convention. Otherwise assume an nD rotate. Angle
        conversion are done either using degrees or radians depending on the
        degrees boolean parameter.
    scale : 1-D array
        A 1-D array of factors to scale each axis by. Scale is broadcast to 1
        in leading dimensions, so that, for example, a scale of [4, 18, 34] in
        3D can be used as a scale of [1, 4, 18, 34] in 4D without modification.
        An empty translation vector implies no scaling.
    shear : n-D array
        An n-D shear matrix.
    """
    n = matrix.shape[0]

    matrix_inv = np.linalg.inv(matrix)

    decomp = np.linalg.cholesky(np.dot(matrix_inv.T, matrix_inv)).T
    scale = np.diag(decomp).copy()
    shears = np.linalg.inv(decomp / scale[:, np.newaxis])

    shear = shears[np.triu(np.ones((n, n)), 1).astype(bool)]
    rotate = np.dot(matrix_inv, np.linalg.inv(decomp))

    if np.linalg.det(rotate) < 0:
        scale[0] *= -1
        decomp[0] *= -1
        rotate = np.dot(matrix_inv, np.linalg.inv(decomp))

    return np.linalg.inv(rotate), np.divide(1, scale), shear

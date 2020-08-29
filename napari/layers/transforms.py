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

    def __init__(self, scale=(1.0,), translate=(0.0,), name=None):
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
    rotation : float, 2-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 2-tuple convert into a 3D rotation matrix, otherwise assume
        an nD rotation. Angle conversion are done either using degrees or
        radians depending on the degrees boolean parameter.
    sheer : n-D array
        An n-D sheer matrix.
    degrees : bool
        Boolean if rotation angles are provided in degrees
    name : string
        A string name for the transform.
    """

    def __init__(
        self,
        scale=(1.0,),
        translate=(0.0,),
        rotation=None,
        sheer=None,
        matrix=None,
        degrees=True,
        name=None,
    ):
        super().__init__(name=name)

        if matrix is None:
            if rotation is None:
                rotation = np.eye(len(scale))
            if sheer is None:
                sheer = np.eye(len(scale))
            matrix = compose_affine_matrix(
                rotation, scale, sheer, degrees=degrees
            )
        else:
            matrix = np.array(matrix)

        self.matrix = matrix
        self.translate = np.array(translate)

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
        return np.atleast_1d(np.squeeze(coords @ matrix.T + translate))

    @property
    def scale(self) -> np.array:
        """Return the scale of the transform."""
        return decompose_affine_matrix(self.matrix)[1]

    @scale.setter
    def scale(self, scale):
        """Set the scale of the transform."""
        R, Z, S = decompose_affine_matrix(self.matrix)
        self.matrix = compose_affine_matrix(R, scale, S)

    @property
    def rotation(self) -> np.array:
        """Return the rotation of the transform."""
        return decompose_affine_matrix(self.matrix)[0]

    @rotation.setter
    def rotation(self, rotation):
        """Set the rotation of the transform."""
        R, Z, S = decompose_affine_matrix(self.matrix)
        self.matrix = compose_affine_matrix(rotation, Z, S)

    @property
    def sheer(self) -> np.array:
        """Return the sheer of the transform."""
        return decompose_affine_matrix(self.matrix)[2]

    @sheer.setter
    def sheer(self, sheer):
        """Set the rotation of the transform."""
        R, Z, S = decompose_affine_matrix(self.matrix)
        self.matrix = compose_affine_matrix(R, Z, sheer)

    @property
    def inverse(self) -> 'Affine':
        """Return the inverse transform."""
        matrix = np.linalg.inv(self.matrix)
        translate = -matrix @ self.translate
        return Affine(matrix=matrix, translate=translate)

    def compose(self, transform: 'Affine') -> 'Affine':
        """Return the composite of this transform and the provided one."""
        matrix = self.matrix @ transform.matrix
        translate = self.translate + self.matrix @ transform.translate
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


def compose_affine_matrix(rotation, scale, sheer, degrees=True) -> np.array:
    """Compose matrix from rotation, sheer, scale."""

    if np.isscalar(rotation):
        # If a scalar is passed assume it is a single rotation angle
        # for a 2D rotation
        if degrees:
            theta = np.deg2rad(rotation)
        else:
            theta = rotation
        rotation_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    elif np.array(rotation).ndim == 1 and len(rotation) == 2:
        # If a 2-tuple is passed assume it is two rotation angles
        # for a 3D rotation
        if degrees:
            theta = np.deg2rad(rotation[0])
            phi = np.deg2rad(rotation[1])
        else:
            theta = rotation[0]
            phi = rotation[1]
        rotation_mat = np.array(
            [
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
                [
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    -np.sin(theta),
                ],
                [-np.sin(phi), np.cos(phi), 0],
            ]
        )
    else:
        # Otherwise assume a full nD rotation matrix has been passed
        rotation_mat = np.array(rotation)

    # Convert a scale vector to an nD diagonal matrix
    scale_mat = np.diag(scale)

    # Assume a full nD sheer matrix has been passed
    sheer_mat = np.array(sheer)

    # Check the dimensionality of the transforms and pad as needed
    n_scale = scale_mat.shape[0]
    n_rotation = rotation_mat.shape[0]
    n_sheer = sheer_mat.shape[0]
    ndim = max(n_scale, n_rotation, n_sheer)

    full_scale = embed_in_identity_matrix(scale_mat, ndim)
    full_rotation = embed_in_identity_matrix(rotation_mat, ndim)
    full_sheer = embed_in_identity_matrix(sheer_mat, ndim)

    return full_scale @ full_rotation @ full_sheer


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
    np.array shape (N, N)
        Larger matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Improper transform matrix {matrix}')

    if matrix.shape[0] == ndim:
        return matrix
    else:
        full_matrix = np.eye(ndim)
        full_matrix[-matrix.shape[0] :, -matrix.shape[1]] = matrix
        return full_matrix


def decompose_affine_matrix(matrix) -> (np.array, np.array, np.array):
    """Decompose the transform matrix into rotation, sheer, scale."""
    RZS = matrix
    ZS = np.linalg.cholesky(np.dot(RZS.T, RZS)).T
    Z = np.diag(ZS).copy()
    shears = ZS / Z[:, np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n, n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
    return R, Z, S

from typing import Sequence

import numpy as np
import toolz as tz

from ...utils.translations import trans
from ..events import EventedList
from .transform_utils import (
    coerce_rotate,
    coerce_scale,
    coerce_shear,
    coerce_translate,
)


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
            raise ValueError(
                trans._('Inverse function was not provided.', deferred=True)
            )

    def compose(self, transform: 'Transform') -> 'Transform':
        """Return the composite of this transform and the provided one."""
        raise ValueError(
            trans._('Transform composition rule not provided', deferred=True)
        )

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
        raise NotImplementedError(
            trans._('Cannot subset arbitrary transforms.', deferred=True)
        )

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
        raise NotImplementedError(
            trans._('Cannot subset arbitrary transforms.', deferred=True)
        )


class TransformChain(EventedList, Transform):
    def __init__(self, transforms=[]):
        super().__init__(
            data=transforms,
            basetype=Transform,
            lookup={str: lambda x: x.name},
        )
        # The above super().__init__() will not call Transform.__init__().
        # For that to work every __init__() called using super() needs to
        # in turn call super().__init__(). So we call it explicitly here.
        Transform.__init__(self)

    def __call__(self, coords):
        return tz.pipe(coords, *self)

    def __newlike__(self, iterable):
        return TransformChain(iterable)

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


class Affine(Transform):
    """n-dimensional affine transformation class.

    The affine transform can be represented as a n+1 dimensional
    transformation matrix in homogeneous coordinates [1]_, an n
    dimensional matrix and a length n translation vector.

    The affine_matrix representation can be used for easy compatibility
    with other libraries that can generate affine transformations.

    Parameters
    ----------
    affine_matrix : n-D array
        (N+1, N+1) affine transformation matrix in homogeneous coordinates [1]_.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1.
    name : string
        A string name for the transform.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates.
    """

    def __init__(
        self,
        affine_matrix,
        name=None,
    ):
        super().__init__(name=name)
        self.affine_matrix = affine_matrix

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        if coords.shape[1] != self.linear_matrix.shape[0]:
            linear_matrix = np.eye(coords.shape[1])
            linear_matrix[
                -self.linear_matrix.shape[0] :, -self.linear_matrix.shape[1] :
            ] = self.linear_matrix
        else:
            linear_matrix = self.linear_matrix
        translate = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.translate)), self.translate)
        )
        return np.atleast_1d(np.squeeze(coords @ linear_matrix.T + translate))

    @property
    def ndim(self) -> int:
        """Dimensionality of the transform."""
        return self.affine_matrix.shape[0] - 1

    def __array__(self, *args, **kwargs):
        """NumPy __array__ protocol to get the affine transform matrix."""
        return self.affine_matrix

    @property
    def inverse(self) -> 'Affine':
        """Return the inverse transform."""
        return Affine(affine_matrix=np.linalg.inv(self.affine_matrix))

    def compose(self, transform: 'Affine') -> 'Affine':
        """Return the composite of this transform and the provided one."""
        affine_matrix = self.affine_matrix @ transform.affine_matrix
        return Affine(affine_matrix=affine_matrix)

    @property
    def linear_matrix(self):
        return self.affine_matrix[:-1, :-1]

    @property
    def translate(self):
        return self.affine_matrix[:-1, -1]

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
        ndim = len(axes)
        affine_matrix = np.eye(ndim + 1)
        affine_matrix[:-1, :-1] = self.linear_matrix[np.ix_(axes, axes)]
        affine_matrix[:-1, -1] = self.translate[axes]
        return Affine(affine_matrix, name=self.name)

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
        n = len(axes) + self.ndim
        not_axes = [i for i in range(n) if i not in axes]
        affine_matrix = np.eye(n + 1)
        affine_matrix[np.ix_(not_axes, not_axes)] = self.linear_matrix
        affine_matrix[not_axes, -1] = self.translate
        return Affine(affine_matrix=affine_matrix, name=self.name)


class CompositeAffine(Transform):
    """n-dimensional affine transformation composed from more basic components.

    Composition is in the following order

    affine = shear * rotate * scale + translate

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
    translate : 1-D array
        A 1-D array of factors to shift each axis by. Translation is broadcast
        to 0 in leading dimensions, so that, for example, a translation of
        [4, 18, 34] in 3D can be used as a translation of [0, 4, 18, 34] in 4D
        without modification. An empty translation vector implies no
        translation.
    name : string
        A string name for the transform.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates.
    """

    def __init__(
        self,
        ndim,
        *,
        translate=None,
        scale=None,
        rotate=None,
        shear=None,
        name=None,
    ):
        super().__init__(name=name)

        self.translate = coerce_translate(ndim, translate)

        self._scale = coerce_scale(ndim, scale)
        self._rotate = (
            np.eye(ndim) if rotate is None else coerce_rotate(rotate)
        )
        self._shear = np.eye(ndim) if shear is None else coerce_shear(shear)

        self._update_linear_matrix()

    def _update_linear_matrix(self):
        self.linear_matrix = self._shear @ self._rotate @ np.diag(self._scale)

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        if coords.shape[1] != self.linear_matrix.shape[0]:
            linear_matrix = np.eye(coords.shape[1])
            linear_matrix[
                -self.linear_matrix.shape[0] :, -self.linear_matrix.shape[1] :
            ] = self.linear_matrix
        else:
            linear_matrix = self.linear_matrix
        translate = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.translate)), self.translate)
        )
        return np.atleast_1d(np.squeeze(coords @ linear_matrix.T + translate))

    @property
    def ndim(self) -> int:
        """Dimensionality of the transform."""
        return self.translate.shape[0]

    @property
    def scale(self) -> np.array:
        """Return the scale of the transform."""
        return self._scale

    @scale.setter
    def scale(self, scale):
        """Set the scale of the transform."""
        self._scale = scale
        self._update_linear_matrix()

    @property
    def rotate(self) -> np.array:
        """Return the rotation of the transform."""
        return self._rotate

    @rotate.setter
    def rotate(self, rotate):
        """Set the rotation of the transform."""
        self._rotate = coerce_rotate(rotate)
        self._update_linear_matrix()

    @property
    def shear(self) -> np.array:
        """Return the shear of the transform."""
        return self._shear

    @shear.setter
    def shear(self, shear):
        """Set the shear of the transform."""
        self._shear = coerce_shear(shear)
        self._update_linear_matrix()

    @property
    def affine_matrix(self) -> np.array:
        """Return the affine matrix for the transform."""
        matrix = np.eye(self.ndim + 1, self.ndim + 1)
        matrix[:-1, :-1] = self.linear_matrix
        matrix[:-1, -1] = self.translate
        return matrix

    def __array__(self, *args, **kwargs):
        """NumPy __array__ protocol to get the affine transform matrix."""
        return self.affine_matrix

    @property
    def inverse(self) -> 'Affine':
        """Return the inverse transform."""
        return Affine(affine_matrix=np.linalg.inv(self.affine_matrix))

    def compose(self, transform: 'Affine') -> 'Affine':
        """Return the composite of this transform and the provided one."""
        affine_matrix = self.affine_matrix @ transform.affine_matrix
        return Affine(affine_matrix=affine_matrix)

    def set_slice(self, axes: Sequence[int]) -> 'CompositeAffine':
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
        return CompositeAffine(
            len(axes),
            scale=self.scale[axes],
            translate=self.translate[axes],
            rotate=self.rotate[np.ix_(axes, axes)],
            shear=self.shear[np.ix_(axes, axes)],
            name=self.name,
        )

    def expand_dims(self, axes: Sequence[int]) -> 'CompositeAffine':
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
        rotate = np.eye(n)
        rotate[np.ix_(not_axes, not_axes)] = self.rotate
        shear = np.eye(n)
        shear[np.ix_(not_axes, not_axes)] = self.shear
        translate = np.zeros(n)
        translate[not_axes] = self.translate
        scale = np.ones(n)
        scale[not_axes] = self.scale
        return CompositeAffine(
            n,
            translate=translate,
            scale=scale,
            rotate=rotate,
            shear=shear,
            name=self.name,
        )

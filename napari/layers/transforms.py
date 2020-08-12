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

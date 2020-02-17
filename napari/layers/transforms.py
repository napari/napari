from __future__ import annotations  # noqa: F407

import toolz as tz
from typing import Sequence
import numpy as np

from ..utils.list import ListModel


class Transform:
    """Base transform class.

    Defaults to the identity transform.

    Parameters
    ----------
    func : callable, Coords -> Coords
        A function converting an NxD array of coordinates to NxD'.
    """

    def __init__(self, func=tz.identity, inverse=None, name=None):
        self.func = func
        self._inverse_func = inverse
        if func is tz.identity:
            self._inverse_func = tz.identity
        self.name == name

    def __call__(self, coords):
        """Transform input coordinates to output."""
        return self.func(coords)

    def set_slice(self, axes: Sequence[int]) -> Transform:
        """Return a transform subset to the visible dimensions."""
        raise NotImplementedError('Cannot subset arbitrary transforms.')

    @property
    def inverse(self):
        if self._inverse_func is not None:
            return Transform(self._inverse_func, self.func)
        else:
            raise ValueError('Inverse function was not provided.')


class TransformChain(Transform, ListModel):
    def __init__(self, transforms=[]):
        super().__init__(basetype=Transform, iterable=transforms, lookup=id)

    def __call__(self, coords):
        return tz.pipe(coords, *self)

    def set_slice(self, axes: Sequence[int]) -> TransformChain:
        return TransformChain([tf.set_slice(axes) for tf in self])


class Translate(Transform):
    """n-dimensional translation (shift) class.

    An empty translation vector implies no translation.

    Translation is broadcast to 0 in leading dimensions, so that, for example,
    a translation of [4, 18, 34] in 3D can be used as a translation of
    [0, 4, 18, 34] in 4D without modification.
    """

    def __init__(self, vector=(0.0,), name='translate'):
        super().__init__(name=name)
        self.vector = np.array(vector)

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        vector = np.concatenate(
            ([0.0] * (coords.shape[1] - len(self.vector)), self.vector)
        )
        return coords + vector

    @property
    def inverse(self):
        return Translate(-self.vector)

    def set_slice(self, axes: Sequence[int]) -> Translate:
        return Translate(self.vector[axes])


class Scale(Transform):
    """n-dimensional scale class.

    An empty scale class implies a scale of 1.
    """

    def __init__(self, scale=(1.0,), name='scale'):
        super().__init__(name=name)
        self.scale = np.array(scale)

    def __call__(self, coords):
        coords = np.atleast_2d(coords)
        if coords.shape[1] > len(self.scale):
            scale = np.concatenate(
                ([1.0] * (coords.shape[1] - len(self.scale)), self.scale)
            )
        return coords * scale

    @property
    def inverse(self):
        return Scale(1 / self.scale)

    def set_slice(self, axes: Sequence[int]) -> Scale:
        return Scale(self.scale[axes])

from enum import auto

import numpy as np

from ...utils.misc import StringEnum
from .colorbars import make_colorbar
from .standardize_color import transform_color


class ColormapInterpolationMode(StringEnum):
    """INTERPOLATION: Interpolation mode for colormaps.

    Selects an interpolation mode for the colormap.
            * linear: colors are defined by linear interpolation between
              colors of neighboring controls points.
            * zero: colors are defined by the value of the color in the
              bin between by neighboring controls points.
    """

    LINEAR = auto()
    ZERO = auto()


class Colormap:
    """Colormap that relates intensity values to colors.

    Parameters
    ----------
    colors : array, shape (N, 4)
        Data used in the colormap.
    controls : array, shape (N,) or (N+1,)
        Control points of the colormap.
    interpolation : str
        Colormap interpolation mode, either 'linear' or
        'zero'. If 'linear', ncontrols = ncolors (one
        color per control point). If 'zero', ncontrols
        = ncolors+1 (one color per bin).
    name : str
        Name of the colormap.
    """

    def __init__(
        self, colors, *, controls=None, interpolation='linear', name='custom'
    ):

        self.name = name
        self.colors = transform_color(colors)
        self._interpolation = ColormapInterpolationMode(interpolation)
        if controls is None:
            n_controls = len(self.colors) + int(
                self._interpolation == ColormapInterpolationMode.ZERO
            )
            self.controls = np.linspace(0, 1, n_controls)
        else:
            self.controls = np.asarray(controls)

    def __iter__(self):
        yield from (self.colors, self.controls, str(self.interpolation))

    def map(self, values):
        values = np.atleast_1d(values)
        if self._interpolation == ColormapInterpolationMode.LINEAR:
            # One color per control point
            cols = [
                np.interp(values, self.controls, self.colors[:, i])
                for i in range(4)
            ]
            cols = np.stack(cols, axis=1)
        elif self._interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            indices = np.clip(
                np.searchsorted(self.controls, values) - 1, 0, len(self.colors)
            )
            cols = self.colors[indices.astype(np.int32)]
        else:
            raise ValueError('Unrecognized Colormap Interpolation Mode')

        return cols

    @property
    def colorbar(self):
        return make_colorbar(self)

    @property
    def interpolation(self):
        return str(self._interpolation)

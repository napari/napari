import numpy as np
from scipy import interpolate
from enum import auto

from .colorbars import make_colorbar
from .standardize_color import transform_color
from ...utils.misc import StringEnum


class ColormapInterpolationMode(StringEnum):
    """INTERPOLATION: Interpolation mode for colormaps."""

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
        self, colors, controls=None, interpolation='linear', name='undefined'
    ):

        self.name = name
        self.colors = transform_color(colors)
        self.interpolation = interpolation
        if controls is None:
            N = len(self.colors) + int(self.interpolation == 'zero')
            self.controls = np.linspace(0, 1, N)
        else:
            self.controls = np.asarray(controls)

    def __iter__(self):
        yield from (self.colors, self.controls, self.interpolation)

    def map(self, values):
        funcs = [
            interpolate.interp1d(
                self.controls, self.colors[:, i], kind=self.interpolation
            )
            for i in range(4)
        ]
        return np.stack([f(values) for f in funcs], axis=1)

    @property
    def colorbar(self):
        return make_colorbar(self)

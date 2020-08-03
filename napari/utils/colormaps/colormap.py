import numpy as np
from scipy import interpolate
from enum import auto

from .colorbars import make_colorbar
from .standardize_color import transform_color
from ...utils.dataclass import dataclass, Property
from ...utils.misc import StringEnum


class ColormapInterpolationMode(StringEnum):
    """INTERPOLATION: Interpolation mode for colormaps."""

    LINEAR = auto()
    ZERO = auto()


@dataclass(events=True, properties=True)
class Colormap:
    """Colormap that relates intensity values to colors.

    Parameters
    ----------
    name : str
        Name of the colormap.
    data : array, shape (N, 4)
        Data used in the colormap.
    controls : array, shape (N,) or (N+1,)
        Control points of the colormap.
    interpolation : str
        Colormap interpolation mode, either 'linear' or
        'zero'. If 'linear', ncontrols = ncolors (one
        color per control point). If 'zero', ncontrols
        = ncolors+1 (one color per bin).
    """

    name: str = 'undefined'
    colors: Property[np.ndarray, None, transform_color] = np.zeros((2, 4))
    controls: Property[np.ndarray, None, np.asarray] = np.empty((0))
    interpolation: ColormapInterpolationMode = 'linear'

    def __post_init__(self):
        if len(self.controls) == 0:
            N = len(self.colors) + int(self.interpolation == 'zero')
            self.controls = np.linspace(0, 1, N)

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

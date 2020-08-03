import numpy as np
from scipy import interpolate

from .colorbars import make_colorbar
from .standardize_color import transform_color
from ...utils.dataclass import dataclass, Property


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
        Colormap interpolation mode, either `linear` or
        `zero`.
    """

    name: str = 'undefined'
    colors: Property[np.ndarray, None, transform_color] = np.zeros((2, 4))
    controls: Property[np.ndarray, None, np.asarray] = np.asarray(
        [0, 1]
    )  # Not yet implemented
    interpolation: str = 'linear'  # Only linear supported

    def map(self, values):
        x = np.linspace(0, 1, len(self.colors))  # Should use control points
        funcs = [
            interpolate.interp1d(x, self.colors[:, i], kind='linear')
            for i in range(4)
        ]
        mapped = np.array(
            [f(values) for f in funcs]
        ).T  # Could probably be nicer
        return mapped

    @property
    def colorbar(self):
        return make_colorbar(self)

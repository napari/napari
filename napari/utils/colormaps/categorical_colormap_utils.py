from dataclasses import dataclass
from itertools import cycle

import numpy as np


@dataclass(eq=False)
class ColorCycle:
    """A dataclass to hold a color cycle for the fallback_colors
    in the CategoricalColormap

    Attributes
    ----------
    values : np.ndarray
        The (Nx4) color array of all colors contained in the color cycle.
    cycle : cycle
        The cycle object that gives fallback colors.
    """

    values: np.ndarray
    cycle: cycle

    def __eq__(self, other):
        if isinstance(other, ColorCycle):
            eq = np.array_equal(self.values, other.values)
        else:
            eq = False

        return eq


def compare_colormap_dicts(cmap_1, cmap_2):

    if len(cmap_1) != len(cmap_2):
        return False
    for k, v in cmap_1.items():
        if k not in cmap_2:
            return False
        if not np.allclose(v, cmap_2[k]):
            return False
    return True

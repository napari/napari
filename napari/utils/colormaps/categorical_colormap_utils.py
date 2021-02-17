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

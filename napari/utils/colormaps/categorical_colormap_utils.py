from dataclasses import dataclass
from itertools import cycle

import numpy as np


@dataclass
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


def compare_color_cycle(cycle_1: ColorCycle, cycle_2: ColorCycle) -> bool:
    """Equality check that returns true if two ColorCycle objects are equal"""
    return np.array_equal(cycle_1.values, cycle_2.values)

from dataclasses import dataclass
from itertools import cycle

import numpy as np


@dataclass
class ColorCycle:
    values: np.ndarray
    cycle: cycle


def compare_color_cycle(cycle_1: ColorCycle, cycle_2: ColorCycle) -> bool:
    return np.array_equal(cycle_1.values, cycle_2.values)

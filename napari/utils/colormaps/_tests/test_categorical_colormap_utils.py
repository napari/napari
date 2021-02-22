from itertools import cycle

import numpy as np

from ..categorical_colormap_utils import ColorCycle


def test_color_cycle():
    color_values = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])
    color_cycle = cycle(color_values)
    cc_1 = ColorCycle(values=color_values, cycle=color_cycle)
    cc_2 = ColorCycle(values=color_values, cycle=color_cycle)

    np.testing.assert_allclose(cc_1.values, color_values)
    assert isinstance(cc_1.cycle, cycle)
    assert cc_1 == cc_2

    other_color_values = np.array([[1, 0, 0, 1], [1, 1, 1, 1]])
    other_color_cycle = cycle(other_color_values)
    cc_3 = ColorCycle(values=other_color_values, cycle=other_color_cycle)
    assert cc_1 != cc_3

    # verify that checking equality against another type works
    assert cc_1 != color_values

"""Units utilities."""

from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pint

# define preferred scale bar values
PREFERRED_VALUES = [
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    50,
    75,
    100,
    125,
    150,
    200,
    500,
    750,
]


@lru_cache(maxsize=1)
def get_unit_registry() -> 'pint.UnitRegistry':
    """Get pint's UnitRegistry.

    Pint greedily imports many libraries, (including dask, xarray, pandas, and
    babel) to check for compatibility.  Some of those libraries may be slow to
    import.  This accessor function should be used (and only when units are
    actually necessary) to avoid incurring a large import time penalty.

    See comment for details:
    https://github.com/napari/napari/pull/2617#issuecomment-827747792
    """
    import pint

    return pint.UnitRegistry()


PREFERRED_TICK_VALUES = [1, 1.5, 2, 2.5, 3, 4, 5, 7.5]


def _generate_ticks(base, exp, min_value, max_value):
    step = Decimal(base) * Decimal(10) ** exp

    # ensure we never go past min and max
    tick_min = np.ceil(min_value / step) * step
    tick_max = np.floor(max_value / step) * step

    # actually generate ticks with the given step
    return np.arange(tick_min, tick_max + step, step).astype(float)


def compute_nice_ticks(
    min_value: float, max_value: float, target_ticks: int = 5
) -> np.ndarray:
    best_ticks = np.empty(0, float)

    if min_value == max_value:
        return best_ticks

    # Decimal needed for small values float imprecision
    min_value = Decimal(min_value)
    max_value = Decimal(max_value)

    span = max_value - min_value
    ideal_step = span / (target_ticks - 1)

    ideal_exponent = int(np.floor(np.log10(ideal_step)))
    exp_candidates = range(ideal_exponent - 1, ideal_exponent + 1)

    for exp in exp_candidates:
        for base in PREFERRED_TICK_VALUES:
            ticks = _generate_ticks(base, exp, min_value, max_value)

            # get number of ticks closer to target_ticks
            if best_ticks is None or abs(len(ticks) - target_ticks) <= abs(
                len(best_ticks) - target_ticks
            ):
                best_ticks = ticks

    return best_ticks

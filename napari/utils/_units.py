"""Units utilities."""
from functools import lru_cache
from typing import TYPE_CHECKING

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

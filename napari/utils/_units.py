"""Units utilities."""
from functools import lru_cache

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
def get_unit_registry():
    """Get pint's UnitRegistry.

    This method is preferred
    """
    import pint

    return pint.UnitRegistry()

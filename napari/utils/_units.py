"""Units utilities"""
import pint

# instantiate default unit registry
UNIT_REG = pint.UnitRegistry()

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

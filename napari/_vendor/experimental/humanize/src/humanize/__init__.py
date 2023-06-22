from .filesize import naturalsize
from .i18n import activate, deactivate
from .number import apnumber, fractional, intcomma, intword, ordinal, scientific
from .time import (
    naturaldate,
    naturalday,
    naturaldelta,
    naturaltime,
    precisedelta,
)

__version__ = VERSION = "2.5.0"


__all__ = [
    "__version__",
    "activate",
    "apnumber",
    "deactivate",
    "fractional",
    "intcomma",
    "intword",
    "naturaldate",
    "naturalday",
    "naturaldelta",
    "naturalsize",
    "naturaltime",
    "ordinal",
    "precisedelta",
    "scientific",
    "VERSION",
]

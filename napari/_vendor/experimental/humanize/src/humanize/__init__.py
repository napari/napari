import pkg_resources
from napari._vendor.experimental.humanize.src.humanize.filesize import naturalsize
from napari._vendor.experimental.humanize.src.humanize.i18n import activate, deactivate
from napari._vendor.experimental.humanize.src.humanize.number import apnumber, fractional, intcomma, intword, ordinal, scientific
from napari._vendor.experimental.humanize.src.humanize.time import (
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

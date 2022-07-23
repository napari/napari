from ._dask_utils import resize_dask_cache
from .colormaps import Colormap
from .info import citation_text, sys_info
from .notebook_display import nbscreenshot
from .progress import progrange, progress

__all__ = [
    'citation_text',
    'Colormap',
    'nbscreenshot',
    'progrange',
    'progress',
    'resize_dask_cache',
    'split_channels',
    'sys_info',
]


def __getattr__(name):
    # prefer to export here, but moving it from layers.utils causes a circular import
    if name != 'split_channels':
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from ..layers.utils.stack_utils import split_channels

    return split_channels

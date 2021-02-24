from .colormaps import Colormap
from .dask_utils import resize_dask_cache
from .info import citation_text, sys_info
from .notebook_display import nbscreenshot

#: dask.cache.Cache, optional : A dask cache for opportunistic caching
#: use :func:`~.resize_dask_cache` to actually register and resize.
dask_cache = None

from .info import sys_info, citation_text
from .dask_utils import resize_dask_cache
from .notebook_display import nbscreenshot

#: dask.cache.Cache, optional : A dask cache for opportunistic caching
#: use :func:`~.resize_dask_cache` to actually register and resize.
dask_cache = None

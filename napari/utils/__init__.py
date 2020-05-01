from .info import sys_info, citation_text
from .misc import resize_dask_cache
from dask.cache import Cache

#: dask.cache.Cache : A dask cache for opportunistic caching
#: use :func:`~.resize_dask_cache` to actually register and resize.
dask_cache = Cache(0)

del Cache

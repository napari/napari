from .info import sys_info, citation_text
from .misc import resize_dask_cache
from dask.cache import Cache

#: dask.cache.Cache, optional : A dask cache for opportunistic caching
dask_cache = Cache(0)  # use resize_dask_cache to actually register and resize.

del Cache

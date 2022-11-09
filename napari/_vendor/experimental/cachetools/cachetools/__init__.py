"""Extensible memoizing collections and decorators."""

from napari._vendor.experimental.cachetools.cachetools.cache import Cache
from napari._vendor.experimental.cachetools.cachetools.decorators import cached, cachedmethod
from napari._vendor.experimental.cachetools.cachetools.lfu import LFUCache
from napari._vendor.experimental.cachetools.cachetools.lru import LRUCache
from napari._vendor.experimental.cachetools.cachetools.rr import RRCache
from napari._vendor.experimental.cachetools.cachetools.ttl import TTLCache

__all__ = (
    'Cache',
    'LFUCache',
    'LRUCache',
    'RRCache',
    'TTLCache',
    'cached',
    'cachedmethod'
)

__version__ = '4.1.1'

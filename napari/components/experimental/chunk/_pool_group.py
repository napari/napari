"""LoaderPoolGroup class.
"""
from __future__ import annotations

import bisect
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, List

from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components.experimental.chunk._pool import (
        DoneCallback,
        LoaderPool,
    )
    from napari.components.experimental.chunk._request import ChunkRequest


class LoaderPoolGroup:
    """Holds the LoaderPools that the ChunkLoader is using.

    Parameters
    ----------
    octree_config : dict
        The full octree config data.

    Attributes
    ----------
    _pools : Dict[int, LoaderPool]
        The mapping from priority to loader pool.
    """

    def __init__(
        self, octree_config: dict, on_done: DoneCallback = None
    ) -> None:
        self._pools = self._create_pools(octree_config, on_done)

    def _create_pools(
        self, octree_config: dict, on_done: DoneCallback
    ) -> Dict[int, LoaderPool]:
        """Return the mapping from priorities to loaders.

        Parameters
        ----------
        octree_config : dict
            Octree configuration data.

        Returns
        -------
        Dict[int, LoaderPool]
            The loader to use for each priority
        """
        from napari.components.experimental.chunk._pool import LoaderPool

        configs = _get_loader_configs(octree_config)

        # Create a LoaderPool for each priority.
        return {
            priority: LoaderPool(config, on_done)
            for (priority, config) in configs.items()
        }

    def get_loader(self, priority) -> LoaderPool:
        """Return the LoaderPool for the given priority.

        Returns
        -------
        LoaderPool
            The LoaderPool for the given priority.
        """
        use_priority = self._get_loader_priority(priority)
        return self._pools[use_priority]

    @lru_cache(maxsize=64)  # noqa: B019
    def _get_loader_priority(self, priority: int) -> int:
        """Return the loader priority to use.

        This method is pretty fast, but since the mapping from priority to
        LoaderPool is static, use lru_cache.
        """
        priority = max(priority, 0)  # No negative priorities.
        keys = sorted(self._pools.keys())
        index = bisect.bisect_left(keys, priority)
        if index < len(keys) and keys[index] == priority:
            return priority  # Exact hit on a pool, so use it.
        return keys[index - 1]  # Use the pool just before the insertion point.

    def load_async(self, request: ChunkRequest) -> None:
        """Load this request asynchronously.

        Parameters
        ----------
        request : ChunkRequest
            The request to load.
        """
        self.get_loader(request.priority).load_async(request)

    def cancel_requests(
        self, should_cancel: Callable[[ChunkRequest], bool]
    ) -> List[ChunkRequest]:
        """Cancel pending requests based on the given filter.

        Parameters
        ----------
        should_cancel : Callable[[ChunkRequest], bool]
            Cancel the request if this returns True.

        Returns
        -------
        List[ChunkRequests]
            The requests that were cancelled, if any.
        """
        cancelled = []
        for pool in self._pools.values():
            cancelled.extend(pool.cancel_requests(should_cancel))
        return cancelled

    def shutdown(self) -> None:
        """Shutdown the pools."""
        for pool in self._pools.values():
            pool.shutdown()


def _get_loader_configs(octree_config) -> Dict[int, dict]:
    """Return dict of loader configs for the octree.

    We merge each loader config with the defaults, so that each loader
    config only needs to specify non-default values.

    Parameters
    ----------
    octree_config : dict
        The octree configuration data.

    Returns
    -------
    Dict[int, dict]
        A dictionary of loader configs.
    """
    try:
        defaults = octree_config['loader_defaults']
    except KeyError as exc:
        raise KeyError(
            trans._(
                "Missing 'loader_defaults' in octree config.",
                deferred=True,
            )
        ) from exc

    try:
        configs = octree_config['octree']['loaders']
    except KeyError:
        # No octree specific loaders were specificed. We we just have one loader
        # with default, zero priority should catch everything.
        return {0: defaults}

    def merge(config: dict) -> dict:
        """Return config merged with the defaults.

        Can't use dict constructor since we have int keys.
        """
        merged = defaults.copy()
        merged.update(config)
        return merged

    # Return merged configs.
    return {int(key): merge(config) for (key, config) in configs.items()}

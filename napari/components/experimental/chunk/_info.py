"""LoadType, LoadStats and LayerInfo.
"""
import logging
import weakref
from enum import Enum

from ....layers.base import Layer
from ._config import async_config
from ._request import ChunkRequest
from ._utils import StatWindow

LOGGER = logging.getLogger("napari.async")


class LoadType(Enum):
    """Tell the ChunkLoader how it should load this layer."""

    AUTO = 0  # Decide based on load speed.
    SYNC = 1  # Always load synchronously.
    ASYNC = 2  # Always load asynchronously.


class LoadStats:
    """Statistics about the recent loads for one layer.

    Attributes
    ----------
    window_ms : StatWindow
        Keeps track of the average load time over the window.
    """

    WINDOW_SIZE = 10  # Only keeps stats for this many loads.

    def __init__(self):
        self.window_ms: StatWindow = StatWindow(self.WINDOW_SIZE)

    def on_load_finished(self, request: ChunkRequest, sync: bool) -> None:
        """Record stats on this request that was just loaded.

        Parameters
        ----------
        request : ChunkRequest
            The request that was just loaded.
        sync : bool
            True if the load was synchronous.
        """
        # Special "load_chunks" timer was the total time for all chunks combined.
        load_ms = request.timers['load_chunks'].duration_ms
        self.window_ms.add(load_ms)


class LayerInfo:
    """Information about one layer the ChunkLoader is tracking.

    Parameters
    ----------
    layer : Layer
        The layer we are loading chunks for.

    Attributes
    ----------
    layer_id : int
        The id of the layer.
    layer_ref : weakref
        Weak reference to the layer.
    load_type : LoadType
        Enum for whether to do auto/sync/async loads.
    auto_sync_ms : int
        If load takes longer than this many milliseconds make it async.
    stats : LoadStats
        Statistics related the loads.

    Notes
    -----
    We store a weak reference because we do not want an in-progress request
    to prevent a layer from being deleted. Meanwhile, once a request has
    finished, we can de-reference the weakref to make sure the layer was
    note deleted during the load process.
    """

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)
        self.load_type: LoadType = LoadType.AUTO
        self.auto_sync_ms = async_config.auto_sync_ms

        self.stats = LoadStats()

    def get_layer(self) -> Layer:
        """Resolve our weakref to get the layer.

        Returns
        -------
        layer : Layer
            The layer for this ChunkRequest.
        """
        layer = self.layer_ref()
        if layer is None:
            LOGGER.debug(
                "LayerInfo.get_layer: layer %d was deleted", self.layer_id
            )
        return layer

    @property
    def loads_fast(self) -> bool:
        """Return True if this layer has been loading very fast."""
        average = self.stats.window_ms.average
        return average is not None and average <= self.auto_sync_ms

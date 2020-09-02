"""LayerInfo class.
"""
import logging
import weakref
from enum import Enum

from ...layers.base import Layer
from ._request import ChunkRequest
from ._utils import StatWindow

LOGGER = logging.getLogger("napari.async")


def _mbits(num_bytes, duration_ms) -> float:
    """Return Mbit/s."""
    mbits = (num_bytes * 8) / (1024 * 1024)
    seconds = duration_ms / 1000
    if seconds == 0:
        return 0
    return mbits / seconds


class LoadType(Enum):
    """How ChunkLoader should load this layer.

    AUTO means let the ChunkLoader decide, it will load the layer sync or async
    depending on how fast it loads. Otherwise we can lock a layer to SYNC
    or ASYNC.
    """

    AUTO = 0
    SYNC = 1
    ASYNC = 2


class LoadCounts:
    """Cumulative stats for a layer."""

    def __init__(self):
        self.loads: int = 0
        self.chunks: int = 0
        self.bytes: int = 0


class LoadStats:
    """Statistics about async/async loads for one layer."""

    # Keep full details of this many recent loads.
    NUM_RECENT_LOADS = 10

    # Window size for timing statistics. We use a simple average over the
    # window so it doesn't jump around as much as using the last value.
    WINDOW_SIZE = 10

    # Consider loads that takes this or less to be "fast", which will lead
    # us to load the layer sync if the type is LoadType.AUTO.
    MAX_FAST_LOAD_MS = 30

    def __init__(self):
        self.counts: LoadCounts = LoadCounts()

        # Keep running averages of load time and size.
        self.window_ms: StatWindow = StatWindow(self.WINDOW_SIZE)
        self.window_bytes: StatWindow = StatWindow(self.WINDOW_SIZE)

        # Keep most recent NUM_RECENT_LOADS loads
        self.recent_loads: list = []

    @property
    def mbits(self) -> float:
        """Return Mbit/second."""
        return _mbits(self.window_bytes.average, self.window_ms.average)

    @property
    def loads_fast(self) -> bool:
        """Return True if this layer has been loading very fast."""
        avg = self.window_ms.average

        # If average is zero there have been no loads yet.
        return avg > 0 and avg <= self.MAX_FAST_LOAD_MS

    @property
    def recent_load_str(self) -> str:
        """Return string describing the sync/async nature of recent loads.

        Returns
        -------
        str
            Return "sync", "async" or "mixed".
        """
        num_sync = num_async = 0
        for load in self.recent_loads:
            if load.sync:
                num_sync += 1
            else:
                num_async += 1

        if num_async == 0:
            return "sync"
        if num_sync == 0:
            return "async"
        return "mixed"

    def on_load_finished(self, request: ChunkRequest, sync: bool):
        """Record stats on this request that was just loaded.

        Parameters
        ----------
        request : ChunkRequest
            The request that was just loaded.
        sync : bool
            True if the load was synchronous.
        """
        # Record the number of loads and chunks.
        self.counts.loads += 1
        self.counts.chunks += request.num_chunks

        # Increment total bytes loaded.
        num_bytes = request.num_bytes
        self.counts.bytes += num_bytes

        # Time to load all chunks.
        load_ms = request.timers['load_chunks'].duration_ms

        # Update our StatWindows.
        self.window_bytes.add(num_bytes)
        self.window_ms.add(load_ms)

        # Add LoadInfo, keep only NUM_RECENT_LOADS of them.
        load_info = LoadInfo(num_bytes, load_ms, sync=sync)
        keep = self.NUM_RECENT_LOADS - 1
        self.recent_loads = self.recent_loads[-keep:] + [load_info]


class LoadInfo:
    """Information about the load of one request.

    Parameters
    ----------
    num_bytes : int
        How big was this load.
    duration_ms : float
        How long did this load take in milliseconds.
    sync : bool
        True if the load was synchronous.
    """

    def __init__(self, num_bytes: int, duration_ms: float, sync: bool):
        self.num_bytes = num_bytes
        self.duration_ms = duration_ms
        self.sync = sync

    @property
    def mbits(self) -> float:
        """Return Mbits/second."""
        return _mbits(self.num_bytes, self.duration_ms)


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

    Notes
    -----
    We store a weak reference because an in-progress request should not prevent
    a layer from being deleted. Meanwhile once a request has finished, we can
    de-reference to make sure the layer still exists.
    """

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)
        self.load_type: LoadType = LoadType.AUTO
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

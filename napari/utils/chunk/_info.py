"""LayerInfo class.
"""
import logging
import weakref
from enum import Enum

import dask.array as da

from ._request import ChunkRequest
from ._utils import StatWindow

LOGGER = logging.getLogger("ChunkLoader")


class LoadCounts:
    def __init__(self):
        self.loads: int = 0
        self.chunks: int = 0
        self.bytes: int = 0


def _get_type_str(data) -> str:
    """Get human readable name for the data's type.

    Returns
    -------
    str
        A string like "ndarray" or "dask".
    """
    data_type = type(data)

    if data_type == list:
        if len(data) == 0:
            return "EMPTY"
        else:
            # Recursively get the type string of the zeroth level.
            return _get_type_str(data[0])

    if data_type == da.Array:
        # Special case this because otherwise data_type.__name__
        # below would just return "Array".
        return "dask"

    # For class numpy.ndarray this returns "ndarray"
    return data_type.__name__


def _mbits(num_bytes, duration_ms) -> float:
    """Return Mbit/s."""
    mbits = (num_bytes * 8) / (1024 * 1024)
    seconds = duration_ms / 1000
    if seconds == 0:
        return 0
    return mbits / seconds


class LoadType(Enum):
    """How ChunkLoader should load this layer."""

    AUTO = 0  # let ChunkLoader decide
    SYNC = 1
    ASYNC = 2


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
    """

    # Keep full details of this many recent loads.
    NUM_RECENT_LOADS = 10

    # Window size for timing statistics. We use a simple average over the
    # window. This will jump around less than the "last load time" although
    # we could do something fancier than average long term.
    WINDOW_SIZE = 10

    # If the average load speed is less than this, we consider the load
    # speed to be "fast" and we'll load the layer synchronously if the load
    # type is LoadType.AUTO.
    MAX_FAST_LOAD_MS = 30

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)
        self.data_type: str = _get_type_str(layer.data)

        self.counts: LoadCounts = LoadCounts()

        # Keep running averages of load time and size.
        self.window_ms: StatWindow = StatWindow(self.WINDOW_SIZE)
        self.window_bytes: StatWindow = StatWindow(self.WINDOW_SIZE)

        # Keep most recent NUM_RECENT_LOADS loads
        self.recent_loads: list = []

        # By default we let ChunkLoader decide, but we can override that.
        self.load_type: LoadType = LoadType.AUTO

    def get_layer(self):
        """Resolve our weakref to get the layer, log if not found.

        Returns
        -------
        layer : Layer
            The layer for this ChunkRequest.
        """
        layer = self.layer_ref()
        if layer is None:
            LOGGER.info(
                "LayerInfo.get_layer: layer %d was deleted", self.layer_id
            )
        return layer

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

    def load_finished(self, request: ChunkRequest, sync: bool) -> None:
        """This ChunkRequest was satisfied, record stats.

        Parameters
        ----------
        request : ChunkRequest
            Record stats related to loading these chunks.
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

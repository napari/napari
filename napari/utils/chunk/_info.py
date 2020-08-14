"""LayerInfo class.
"""
import logging
import weakref
from enum import Enum

import dask.array as da

from ._request import ChunkRequest
from ._utils import StatWindow

LOGGER = logging.getLogger("ChunkLoader")


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

    DEFAULT = 0  # let ChunkLoader decide
    SYNC = 1
    ASYNC = 2


class LoadInfo:
    """Information about one load.

    """

    def __init__(self, num_bytes, duration_ms):
        self.num_bytes = num_bytes
        self.duration_ms = duration_ms

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

    def __init__(self, layer):
        self.layer_id: int = id(layer)
        self.layer_ref: weakref.ReferenceType = weakref.ref(layer)
        self.data_type: str = _get_type_str(layer.data)

        self.num_loads: int = 0
        self.num_chunks: int = 0
        self.num_bytes: int = 0

        # Keep running averages of load time and size.
        self.window_ms: StatWindow = StatWindow(self.WINDOW_SIZE)
        self.window_bytes: StatWindow = StatWindow(self.WINDOW_SIZE)

        # Keep most recent NUM_RECENT_LOADS loads
        self.recent_loads: list = []

        # By default we let ChunkLoader decide, but we can override that.
        self.load_type: LoadType = LoadType.DEFAULT

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

    def load_finished(self, request: ChunkRequest) -> None:
        """This ChunkRequest was satisfied, record stats.

        Parameters
        ----------
        request : ChunkRequest
            Record stats related to loading these chunks.
        """
        # Record the number of loads and chunks.
        self.num_loads += 1
        self.num_chunks += request.num_chunks

        # Increment total bytes loaded.
        num_bytes = request.num_bytes
        self.num_bytes += num_bytes

        # Update our StatWindows.
        load_ms = request.timers['load_chunks'].duration_ms
        self.window_ms.add(load_ms)
        self.window_bytes.add(num_bytes)

        # Keep at most self.NUM_RECENT_LOADS recent loads.
        load_info = LoadInfo(num_bytes, load_ms)
        keep = self.NUM_RECENT_LOADS - 1
        self.recent_loads = self.recent_loads[-keep:] + [load_info]

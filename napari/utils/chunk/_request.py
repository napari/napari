"""ChunkRequest is used to ask the ChunkLoader to load a chunk.
"""
import time
from typing import Tuple, Union

from ...types import ArrayLike

from ...utils.perf import perf_counter_ns, PerfEvent, timers

# We convert slices to tuple for hashing.
SliceTuple = Tuple[int, int, int]


class ChunkRequest:
    """A request asking the ChunkLoader to load an array.

    Parameters
    ----------
    layer_id : int
        Python id() for the Layer requesting the chunk.
    data_id : int
        Python id() for the Layer._data requesting the chunk.
    indices
        The tuple of slices index into the data.
    array : ArrayLike
        Load the data from this array.

    Attributes
    ----------
    layer_ref : weakref
        Reference to the layer that submitted the request.
    data_id : int
        Python id() of the data in the layer.
    """

    def __init__(self, layer, indices, array: ArrayLike):
        self.layer_id = id(layer)
        self.data_id = id(layer.data)
        self.indices = indices
        self.array = array
        self.delay_seconds = 0

        # Slice objects are not hashable, so turn them into tuples.
        indices_tuple = tuple(_index_to_tuple(x) for x in self.indices)

        # Key is data_id + indices as a tuples.
        self.key = tuple([self.data_id, indices_tuple])

        # Worker process will fill this is then it processes the request.
        self.pid = None

        # If worker is in another process its "timers" object is not the
        # one in the main process. So store up perf events here and
        # submit them back in the main process.
        self.perf_events = []

    def start_timer(self):
        """Start timer timing the array load.
        """
        self.start_seconds = time.time()

    def end_timer(self):
        """End timer and record the perf event.
        """
        self.end_seconds = time.time()

    def add_perf_event(self):
        """Add perf event for this request.
        """
        # We use time.time() in case the request is being run in a separate
        # process, because perf_counter_ns() is not always synchronized
        # across processes/cpus.
        #
        # Since chunk request are "long" time.time() is accurate enough
        # and we convert back to perf_counter_ns here.
        if timers:  # if using perfmon
            delta_ns = perf_counter_ns() - (time.time() * 1e9)
            start_ns = self.start_seconds * 1e9 + delta_ns
            end_ns = self.end_seconds * 1e9 + delta_ns
            timers.add_event(
                PerfEvent("ChunkRequest", start_ns, end_ns, pid=self.pid)
            )


def _index_to_tuple(index: Union[int, slice]) -> Union[int, SliceTuple]:
    """Get hashable object for the given index.

    Slice is not hashable so we convert slices to tuples.

    Parameters
    ----------
    index
        Integer index or a slice.

    Returns
    -------
    Union[int, SliceTuple]
        Hashable object that can be used for the index.
    """
    if isinstance(index, slice):
        return (index.start, index.stop, index.step)
    return index

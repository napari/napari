"""ChunkRequest is used to ask the ChunkLoader to load a chunk.
"""
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

    def start_timer(self):
        self.start_ns = perf_counter_ns()

    def end_timer(self):
        self.end_ns = perf_counter_ns()
        if timers is not None:
            event = PerfEvent(
                "ChunkRequest", self.start_ns, self.end_ns, pid=self.pid
            )
            timers.add_event(event)


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

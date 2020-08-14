"""ChunkRequest is used to ask the ChunkLoader to load chunks.
"""
import contextlib
import logging
import os
import threading
import time
from typing import List, Tuple, Union

import numpy as np

from ...types import ArrayLike, Dict
from ...utils.perf import PerfEvent, perf_counter_ns, timers
from ._utils import get_data_id

LOGGER = logging.getLogger("ChunkLoader")

# We convert slices to tuple for hashing.
SliceTuple = Tuple[int, int, int]

LayerData = Union[ArrayLike, List[ArrayLike]]


class TimeSpan:
    """A span of time in seconds.

    Parameters
    ----------
    start_seconds : float
        Start of the time span.
    end_seconds : float
        End of the time span.
    """

    def __init__(self, start_seconds: float, end_seconds: float, args: dict):
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.args = args

    @property
    def duration_seconds(self) -> float:
        """Return duration of the time span.

        Returns
        -------
        float
            Duration of the time span in seconds.
        """
        return self.end_seconds - self.start_seconds

    @property
    def duration_ms(self) -> float:
        """Return duration of the time span.

        Returns
        -------
        float
            Duration of the time span in milliseconds.
        """
        return self.duration_seconds * 1000


class ChunkKey:
    """The key which a ChunkRequest will load.

    Parameters
    ----------
    layer : Layer
        The layer to load data for.
    indices : ?
        The indices to load from the layer
    """

    def __init__(self, layer, indices):
        self.layer_id = id(layer)
        self.data_id = get_data_id(layer)
        self.data_level = layer._data_level

        # Slice objects are not hashable, so turn them into tuples.
        self.indices = tuple(_index_to_tuple(x) for x in indices)

        # All together as one tuple for easy comparison.
        self.key = (self.layer_id, self.data_id, self.data_level, self.indices)

    def __str__(self):
        return (
            f"layer_id={self.layer_id} data_id={self.data_id} "
            f"data_level={self.data_level} indices={self.indices}"
        )

    def __eq__(self, other):
        return self.key == other.key


class ChunkRequest:
    """A request asking the ChunkLoader to load one or more arrays.

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
    load_seconds : float
        Delay for this long during the load portion.
    """

    def __init__(self, key: ChunkKey, chunks: Dict[str, ArrayLike]):
        # Make sure chunks is str->array as expected.
        for chunk_key, array in chunks.items():
            assert isinstance(chunk_key, str)
            assert array is not None

        self.key = key
        self.chunks = chunks

        # No delay by default, ChunkLoader.load_chunk() will set this if desired.
        self.load_seconds = 0

        # Worker will fill these in with the process and thread of the
        # worker that handles the request.
        self.process_id = None
        self.thread_id = None

        # Record how long each load took. We do this because if we are in
        # a worker process we cannot submit perf timers like normal. We
        # submit them in add_perf_events().
        self.timers: Dict[str, TimeSpan] = {}

    @property
    def num_chunks(self) -> int:
        """Return the number of chunks in this request."""
        return len(self.chunks)

    @property
    def num_bytes(self) -> int:
        """Return the number of bytes that were loaded."""
        return sum(array.nbytes for array in self.chunks.values())

    @property
    def in_memory(self) -> bool:
        """True if all chunks are ndarrays."""
        for array in self.chunks.values():
            if not isinstance(array, np.ndarray):
                return False
        return True

    @contextlib.contextmanager
    def time_block(self, timer_name: str):
        """Like a perf timers but using time.time().

        perf_counter_ns() is not necessarily synchronized across processes
        but time.time() is okay. Since loads are relatively slow probably
        using time.time() is accurate enough.

        Parameters
        ----------
        timer_name : str
            The name of the timer.
        """
        start_seconds = time.time()
        args = {}
        yield args
        end_seconds = time.time()
        self.timers[timer_name] = TimeSpan(start_seconds, end_seconds, args)

    def load_chunks(self):
        """Load all of our chunks now in this thread."""
        self.process_id = os.getpid()
        self.thread_id = threading.get_ident()

        # Time the loading of all the chunks.
        with self.time_block("load_chunks"):

            # Simulate latency.
            if self.load_seconds > 0:
                time.sleep(self.load_seconds)

            # Time the load of every chunk.
            for key, array in self.chunks.items():
                with self.time_block(key) as args:
                    loaded_array = np.asarray(array)
                    self.chunks[key] = loaded_array

                    # Arguments for this PerfEvent.
                    args['shape'] = loaded_array.shape
                    args['nbytes'] = loaded_array.nbytes

    def add_perf_events(self):
        """Add perf events for this request.

        This should be called in the main process after the chunks were
        loaded in the worker. All this effort is not necessary with
        threads, but we do it the same way for processes or threads
        since it will work fine with either.
        """
        if not timers:
            return  # we're not using perfmon

        # Convert from time.time() to perf_counter_ns()
        delta_ns = perf_counter_ns() - (time.time() * 1e9)

        # Add a PerfEvent for each of our timers.
        for name, time_span in self.timers.items():

            # From seconds to nanoseconds
            start_ns = time_span.start_seconds * 1e9 + delta_ns
            end_ns = time_span.end_seconds * 1e9 + delta_ns

            # Add the PerfEvent
            timers.add_event(
                PerfEvent(
                    name,
                    start_ns,
                    end_ns,
                    process_id=self.process_id,
                    thread_id=self.thread_id,
                    **time_span.args,
                )
            )

    def transpose_chunks(self, order):
        """Transpose all our chunks.

        Parameters
        ----------
        order
            Transpose the chunks with this order.
        """
        for key, array in self.chunks.items():
            self.chunks[key] = array.transpose(order)

    @property
    def image(self):
        """The image chunk if we have one or None.
        """
        return self.chunks.get('image')

    @property
    def thumbnail_source(self):
        """The chunk to use as the thumbnail_source or None.
        """
        try:
            return self.chunks['thumbnail_source']
        except KeyError:
            # For single-scale we use the image as the thumbnail_source.
            return self.chunks.get('image')

    def is_compatible(self, layer) -> bool:
        """Return True if the given data is compatible with this request.

        Parameters
        ----------
        data : LayerData
            Compare this data to the data_id in the request.
        """
        other_data_id = get_data_id(layer)
        compatible = self.key.data_id == other_data_id

        if not compatible:
            LOGGER.warn(
                "ChunkRequest data_id=%d does not match %d",
                self.data_id,
                other_data_id,
            )

        return compatible


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

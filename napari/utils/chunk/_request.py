"""ChunkRequest is used to ask the ChunkLoader to load chunks.
"""
from collections import namedtuple
import contextlib
import logging
import os
import threading
import time
from typing import Tuple, Union

import numpy as np

from ...types import ArrayLike, Dict

from ...utils.perf import perf_counter_ns, PerfEvent, timers

LOGGER = logging.getLogger("ChunkLoader")

# We convert slices to tuple for hashing.
SliceTuple = Tuple[int, int, int]

TimeSpan = namedtuple('TimeSpan', "start_seconds end_seconds")


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
    """

    def __init__(self, layer, indices, chunks: Dict[str, ArrayLike]):
        self.layer_id = id(layer)
        self.data_id = id(layer.data)
        self.indices = indices
        self.chunks = chunks
        self.delay_seconds = 0

        # Slice objects are not hashable, so turn them into tuples.
        indices_tuple = tuple(_index_to_tuple(x) for x in self.indices)

        # Key is data_id + indices as a tuples.
        self.key = tuple([self.data_id, indices_tuple])

        # Worker will fill these in, so they are correct for the worker.
        self.process_id = None
        self.thread_id = None

        # Record how long each load took. We do this because if we are in
        # a worker process we cannot submit perf timers like normal. We
        # submit them in add_perf_events().
        self.time_blocks: Dict[str, TimeSpan] = {}

    @property
    def in_memory(self):
        """False if any chunk request are not ndarrays.
        """
        for array in self.chunks.values():
            if not isinstance(array, np.ndarray):
                return False
        return True

    @contextlib.contextmanager
    def time_block(self, timer_name: str):
        """Like a perf timers but using time.time().

        perf_coutner_ns() is not necessarily synchronized across processes
        but time.time() is okay. Since loads are relatively slow
        time.time() is accurate enough.

        Parameters
        ----------
        timer_name : str
            The name of the timer.
        """
        start_seconds = time.time()
        yield
        end_seconds = time.time()
        self.time_blocks[timer_name] = TimeSpan(start_seconds, end_seconds)

    def load_chunks_gui(self):
        """Load ndarray chunks immediately in the GUI thread.

        This is the same as load_chunks() but we don't bother to time it
        because these are already ndarrays.
        """
        for key, array in self.chunks.items():
            self.chunks[key] = np.asarray(array)

    def load_chunks(self):
        """Load all of our chunks."""
        # Record these now, since we are in the worker.
        self.process_id = os.getpid()
        self.thread_id = threading.get_ident()

        # Time the loading of all the chunks.
        with self.time_block("load_chunks"):

            # Delay if requested.
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

            # Time the load of every chunk.
            for key, array in self.chunks.items():
                with self.time_block(key):
                    self.chunks[key] = np.asarray(array)

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

        # Add a PerfEvent for each of our time_blocks.
        for name, time_span in self.time_blocks.items():

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
                )
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

"""ChunkKey and ChunkRequest classes.
"""
import contextlib
import logging
import time
from typing import Optional, Tuple

import numpy as np

from ....types import ArrayLike, Dict
from ....utils.perf import PerfEvent, block_timer
from .layer_key import LayerKey

LOGGER = logging.getLogger("napari.loader")

# We convert slices to tuple for hashing.
SliceTuple = Tuple[Optional[int], Optional[int], Optional[int]]


class ChunkKey:
    """The key for one single ChunkRequest.

    Parameters
    ----------
    layer : Layer
        The layer which is loading the chunk.
    indices : Indices
        The indices of the layer we are loading.

    Attributes
    ----------
    layer_key : LayerKey
        The layer specific parts of the key.
    key : Tuple
        The combined key, everything hashed together.
    """

    def __init__(self, layer_key: LayerKey):
        self.layer_key = layer_key
        self.key = hash(self.get_hash_values())

    def get_hash_values(self):
        return self.layer_key.get_hash_values()

    def __str__(self):
        layer_key = self.layer_key
        return (
            f"layer_id={layer_key.layer_id} data_id={layer_key.data_id} "
            f"data_level={layer_key.data_level} indices={layer_key.indices}"
        )

    def __eq__(self, other):
        return self.key == other.key

    @property
    def location(self):
        # Derived OctreeChunkKey has a location, we don't, but we could maybe
        # print the indices or something here?
        return "none"


class ChunkRequest:
    """A request asking the ChunkLoader to load data.

    Parameters
    ----------
    key : ChunkKey
        The key of the request.
    chunks : Dict[str, ArrayLike]
        One or more arrays that we need to load.

    Attributes
    ----------
    key : ChunkKey
        The key of the request.
    chunks : Dict[str, ArrayLike]
        One or more arrays that we need to load.
    create_time : float
        The time the request was created.
    _timers : Dict[str, PerfEvent]
        Timing information about chunk load time.
    """

    def __init__(self, key: ChunkKey, chunks: Dict[str, ArrayLike]):
        # Make sure chunks dict is what we expect.
        for chunk_key, array in chunks.items():
            assert isinstance(chunk_key, str)
            assert array is not None

        self.key = key
        self.chunks = chunks

        # TODO_OCTREE: Use time.time() or perf_counter_ns here?
        self.create_time = time.time()
        self._timers: Dict[str, PerfEvent] = {}

    @property
    def elapsed_ms(self) -> float:
        """The total time elapsed since the request was created.

        Return
        ------
        float
            The total time elapsed since the chunk was created.
        """
        return (time.time() - self.create_time) * 1000

    @property
    def load_ms(self) -> float:
        """The total time it took to load all chunks.

        Return
        ------
        float
            The total time it took to return all chunks.
        """
        return sum(
            perf_timer.duration_ms for perf_timer in self._timers.values()
        )

    @property
    def data_id(self) -> int:
        """Return the data_id for this request.

        Return
        ------
        int
            The data_id for this request.
        """
        return self.key.layer_key.data_id

    @property
    def num_chunks(self) -> int:
        """Return the number of chunks in this request.

        Return
        ------
        int
            The number of chunks in this request.
        """
        return len(self.chunks)

    @property
    def num_bytes(self) -> int:
        """Return the number of bytes that were loaded.

        Return
        ------
        int
            The number of bytes that were loaded.
        """
        return sum(array.nbytes for array in self.chunks.values())

    @property
    def in_memory(self) -> bool:
        """Return True if all chunks are ndarrays.

        Return
        ------
        bool
            True if all chunks are ndarrays.
        """
        return all(isinstance(x, np.ndarray) for x in self.chunks.values())

    @contextlib.contextmanager
    def _chunk_timer(self, name):
        """Time a block of code and save the PerfEvent in self._timers.

        We want to time our loads whether perfmon is enabled or not, since
        the auto-async feature needs to work in all cases.

        Parameters
        ----------
        name : str
            The name of the timer.

        Yields
        ------
        PerfEvent
            The timing event for the block.
        """
        with block_timer(name) as event:
            yield event
        self._timers[name] = event

    def load_chunks(self):
        """Load all of our chunks now in this thread.

        We time the overall load with the special name "load_chunks" and then
        we time each chunk as it loads, using it's array name as the key.
        """
        for key, array in self.chunks.items():
            with self._chunk_timer(key):
                loaded_array = np.asarray(array)
                self.chunks[key] = loaded_array

    def transpose_chunks(self, order: tuple) -> None:
        """Transpose all our chunks.

        Parameters
        ----------
        order
            Transpose the chunks into this order.
        """
        for key, array in self.chunks.items():
            self.chunks[key] = array.transpose(order)

    @property
    def image(self) -> Optional[ArrayLike]:
        """The image chunk or None.

        Returns
        -------
        Optional[ArrayLike]
            The image chunk or None if we don't have one.
        """
        return self.chunks.get('image')

    @property
    def thumbnail_source(self):
        """The chunk to use as the thumbnail_source or None.

        Returns
        -------
        Optional[ArrayLike]
            The thumbnail_source chunk or None if we don't have one.
        """
        try:
            return self.chunks['thumbnail_source']
        except KeyError:
            # No thumbnail_source so return the image instead. For single-scale
            # we use the image as the thumbnail_source.
            return self.chunks.get('image')

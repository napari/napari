"""LayerRef, ChunkLocation and ChunkRequest classes.
"""
from __future__ import annotations

import contextlib
import logging
import time
import weakref
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple

import numpy as np

from napari.utils.perf import PerfEvent, block_timer

LOGGER = logging.getLogger("napari.loader")

if TYPE_CHECKING:
    from napari.types import ArrayLike

# We convert slices to tuple for hashing.
SliceTuple = Tuple[Optional[int], Optional[int], Optional[int]]


class LayerRef(NamedTuple):
    """A weakref to a layer and its id."""

    layer_id: int
    layer_ref: weakref.ReferenceType

    @property
    def layer(self):
        return self.layer_ref()

    @classmethod
    def from_layer(cls, layer):
        return cls(id(layer), weakref.ref(layer))


class ChunkLocation:
    """Location of the chunk.

    ChunkLocation is the base class for two classes:
        ImageLocation - pre-octree async loading
        OctreeLocation - octree async loading

    Parameters
    ----------
    layer_id : int
        The id of the layer containing the chunks.
    layer_ref : weakref.ReferenceType
        Weak reference to the layer.
    """

    def __init__(self, layer_ref: LayerRef):
        self.layer_ref = layer_ref

    def __eq__(self, other) -> bool:
        return self.layer_ref.layer_id == other.layer_ref.layer_id

    @property
    def layer_id(self) -> int:
        return self.layer_ref.layer_id

    @classmethod
    def from_layer(cls, layer):
        return cls(LayerRef.from_layer(layer))


class OctreeLocation(ChunkLocation):
    """Location of one chunk within the octree.

    Parameters
    ----------
    layer_ref : LayerRef
        Referen to the layer this location is in.
    slice_id : int
        The id of the OctreeSlice we are in.
    level_index : int
        The octree level index.
    row : int
        The chunk row.
    col : int
        The chunk col.
    """

    def __init__(
        self,
        layer_ref: LayerRef,
        slice_id: int,
        level_index: int,
        row: int,
        col: int,
    ):
        super().__init__(layer_ref)
        self.slice_id: int = slice_id
        self.level_index: int = level_index
        self.row: int = row
        self.col: int = col

    def __str__(self):
        return f"location=({self.level_index}, {self.row}, {self.col}) "

    def __eq__(self, other) -> bool:
        return (
            self.slice_id == other.slice_id
            and self.level_index == other.level_index
            and self.row == other.row
            and self.col == other.col
        )

    def __hash__(self) -> int:
        return hash((self.slice_id, self.level_index, self.row, self.col))


class ChunkRequest:
    """A request asking the ChunkLoader to load data.

    Parameters
    ----------
    location : ChunkLocation
        The location of this chunk. Probably a class derived from ChunkLocation
        such as ImageLocation or OctreeLocation.
    chunks : Dict[str, ArrayLike]
        One or more arrays that we need to load.

    Attributes
    ----------
    location : ChunkLocation
        The location of the chunks.
    chunks : Dict[str, ArrayLike]
        One or more arrays that we need to load.
    create_time : float
        The time the request was created.
    _timers : Dict[str, PerfEvent]
        Timing information about chunk load time.
    """

    def __init__(
        self,
        location: ChunkLocation,
        chunks: Dict[str, ArrayLike],
        priority: int = 0,
    ):
        # Make sure chunks dict is valid.
        for chunk_key, array in chunks.items():
            assert isinstance(chunk_key, str)
            assert array is not None

        self.location = location
        self.chunks = chunks

        self.create_time = time.time()
        self._timers: Dict[str, PerfEvent] = {}

        self.priority = priority

    @property
    def elapsed_ms(self) -> float:
        """The total time elapsed since the request was created.

        Returns
        -------
        float
            The total time elapsed since the chunk was created.
        """
        return (time.time() - self.create_time) * 1000

    @property
    def load_ms(self) -> float:
        """The total time it took to load all chunks.

        Returns
        -------
        float
            The total time it took to return all chunks.
        """
        return sum(
            perf_timer.duration_ms for perf_timer in self._timers.values()
        )

    @property
    def num_chunks(self) -> int:
        """The number of chunks in this request.

        Returns
        -------
        int
            The number of chunks in this request.
        """
        return len(self.chunks)

    @property
    def num_bytes(self) -> int:
        """The number of bytes that were loaded.

        Returns
        -------
        int
            The number of bytes that were loaded.
        """
        return sum(array.nbytes for array in self.chunks.values())

    @property
    def in_memory(self) -> bool:
        """True if all chunks are ndarrays.

        Returns
        -------
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

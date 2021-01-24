"""OctreeChunkGeom, OctreeLocation and OctreeChunk classes.
"""
import logging
from typing import List, NamedTuple

import numpy as np

from ....components.experimental.chunk import ChunkLocation, LayerRef
from ....types import ArrayLike

LOGGER = logging.getLogger("napari.octree")


class OctreeChunkGeom(NamedTuple):
    """Position and size of the chunk, for rendering."""

    pos: np.ndarray
    size: np.ndarray


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


class OctreeChunk:
    """A geographically meaningful portion of the full 2D or 3D image.

    For 2D images a chunk is a "tile". It's a 2D square region of pixels
    which are part of the full 2D image.

    For level 0 of the octree, the pixels are 1:1 identical to the full
    image. For level 1 or greater the pixels are downsampled from the full
    resolution image.

    For 3D, not yet implemented, a chunk is a sub-volume. Again for level 0
    the voxels are at the full resolution of the full image, but for other
    levels the voxels are downsampled.

    The highest level of the tree contains a single chunk which depicts the
    entire image, whether 2D or 3D.

    Parameters
    ----------
    data : ArrayLike
        The data to draw for this chunk.
    location : OctreeLocation
        The location of this chunk, including the level_index, row, col.
    geom : OctreeChunkGeom
        The position and size of the chunk.

    Attributes
    ----------
    _orig_data : ArrayLike
        The original unloaded data that we use to implement OctreeChunk.clear().
    loading : bool
        If True the chunk has been queued to be loaded.
    """

    def __init__(
        self, data: ArrayLike, location: OctreeLocation, geom: OctreeChunkGeom
    ):
        self._data = data
        self.location = location
        self.geom = geom

        self.loading = False  # Are we currently being loaded.
        self._orig_data = data  # For clear(), this might go away.

    def __str__(self):
        return f"{self.location}"

    def __hash__(self):
        return hash(self.location)

    @property
    def data(self) -> ArrayLike:
        """Return the data associated with this chunk.

        Before the chunk has been loaded this might be an ndarray or it
        might be Dask array or other array-like object. After the chunk has
        been loaded it will always be an ndarray. By "loaded" we mean the
        bytes are in memory and ready to be drawn.
        """
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """Set the new data for this chunk.

        We set the data after a chunk has been loaded.

        Parameters
        ----------
        data : np.ndarray
            The new data for the chunk.
        """
        # An ndarray means the data is actual bytes in memory.
        assert isinstance(data, np.ndarray)

        # Assign and note the loading process has now finished.
        self._data = data
        self.loading = False

    @property
    def in_memory(self) -> bool:
        """Return True if the data is fully in memory.

        Returns
        -------
        bool
            True if data is fully in memory.
        """
        return isinstance(self.data, np.ndarray)

    @property
    def needs_load(self) -> bool:
        """Return true if this chunk needs to loaded.

        An unloaded chunk's data might be a Dask or similar deferred array.
        A loaded chunk's data is always an ndarray.

        Returns
        -------
        True if the chunk needs to be loaded.
        """
        return not self.in_memory and not self.loading

    def clear(self) -> None:
        """Clear out our loaded data, return to the original.

        This is only done when running without the cache, so that we reload
        the data again. With computation the loaded data might be different
        each time, so we need to do it each time.

        TODO_OCTREE: Can we get rid of clear() if we always nuke the
        contents of every chunk as soon as it's no longer in view? If we do
        that the same chunk will have to be re-created if it comes into
        view a second time, but in most cases the data itself should be
        cached so that shouldn't take long.
        """
        self._data = self._orig_data
        self.loading = False


def log_chunks(
    label: str,
    chunks: List[OctreeChunk],
    location: OctreeLocation = None,
) -> None:
    """Log the given chunks with an intro header message.

    Parameters
    ----------
    label : str
        Prefix the log message with this label.
    chunks : List[OctreeChunk]
        The chunks to log.
    location : Optional[OctreeLocation]
        Append the log message with this location.
    """
    if location is None:
        LOGGER.debug("%s has %d chunks:", label, len(chunks))
    else:
        LOGGER.debug("%s has %d chunks at %s", label, len(chunks), location)
    for i, chunk in enumerate(chunks):
        LOGGER.debug(
            "Chunk %d %s in_memory=%d loading=%d",
            i,
            chunk.location,
            chunk.in_memory,
            chunk.loading,
        )

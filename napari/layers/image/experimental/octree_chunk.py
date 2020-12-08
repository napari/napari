"""OctreeChunk class
"""
from typing import NamedTuple

import numpy as np

from ....components.experimental.chunk import ChunkKey, LayerKey
from ....types import ArrayLike


class OctreeChunkGeom(NamedTuple):
    """Position and scale of the chunk, for rendering.

    Stored in the OctreeChunk so that we calculate them just once
    at OctreeChunk creation time.
    """

    pos: np.ndarray
    scale: np.ndarray


class OctreeLocation(NamedTuple):
    """Location of one chunk within the octree.

    Part of the OctreeChunkKey to uniquely identify a chunk.
    """

    slice_id: int
    level_index: int
    row: int
    col: int

    def __str__(self):
        return (
            f"location =({self.level_index}, {self.row}, {self.col}) "
            # f"slice={self.slice_id} id={id(self)}"
        )

    @classmethod
    def create_null(cls):
        """Create null location that points to nothing."""
        return cls(0, 0, 0, 0, np.zeros(0), np.zeros(0))


class OctreeChunkKey(ChunkKey):
    """A ChunkKey plus some octree specific fields.

    The ChunkLoader uses ChunkKey to identify chunks, for example for
    caching or just tracking what has been loaded.

    Parameters
    ----------
    layer : Layer
        The OctreeImage layer.
    indices : Tuple[Optional[slice], ...]
        The indices of the image we are viewing.
    location : OctreeLocation
        The location of the chunk within the octree.
    """

    def __init__(
        self, layer_key: LayerKey, location: OctreeLocation,
    ):
        self.location = location
        super().__init__(layer_key)

    def _get_hash_values(self):
        # TODO_OCTREE: can't we just hash in the parent's hashed key with
        # our additional values? Probably, but we do it from scratch here.
        parent = super()._get_hash_values()
        return parent + (self.location,)


class OctreeChunk:
    """A geographically meaningful portion of the full 2D or 3D image.

    For 2D images a chunk is a "tile". It's a 2D square region of pixels
    which are part of the full 2D image.

    If it's in level 0 of the octree, the pixels are 1:1 identical to the
    full image. If it's in level 1 or greater the pixels are downsampled
    from the full resolution image.

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
        The x, y coordinates and scale of the chunk.

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
    def key(self) -> OctreeChunkKey:
        """The unique key for this chunk.

        TODO_OCTREE: Switch to __hash__? Tried __hash__ a while ago and ran
        into problems, but maybe try again.
        """
        return (
            self.geom.pos[0],
            self.geom.pos[1],
            self.location.level_index,
        )

    @property
    def in_memory(self) -> bool:
        """Return True if the data is fully in memory.

        Return
        ------
        bool
            True if data is fully in memory.
        """
        return isinstance(self.data, np.ndarray)

    @property
    def needs_load(self) -> bool:
        """Return true if this chunk needs to loaded.

        An unloaded chunk's data might be a Dask or similar deferred array.
        A loaded chunk's data is always an ndarray.

        Return
        ------
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

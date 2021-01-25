"""ImageLocation class.

ImageLocation is the pre-octree Image class's ChunkLocation. When we request
that the ChunkLoader load a chunk, we use this ChunkLocation to identify
the chunk we are requesting and once it's loaded.
"""
import numpy as np

from ....components.experimental.chunk import ChunkLocation, LayerRef
from ....layers import Layer


def get_data_id(data) -> int:
    """Return the data_id to use for this layer.

    Parameters
    ----------
    data
        Get the data_id for this data.

    Notes
    -----
    We use data_id rather than just the layer_id, because if someone
    changes the data out from under a layer, we do not want to use the
    wrong chunks.
    """
    if isinstance(data, list):
        assert data  # data should not be empty for image layers.
        return id(data[0])  # Just use the ID from the 0'th layer.

    return id(data)  # Not a list, just use it.


class ImageLocation(ChunkLocation):
    """The hashable location of a chunk within an image layer.

    Attributes
    ----------
    data_id : int
        The id of the data in the layer.
    data_level : int
        The level in the data (for multi-scale).
    indices
        The indices of the slice.
    """

    def __init__(self, layer: Layer, indices):
        super().__init__(LayerRef.from_layer(layer))
        self.data_id: int = get_data_id(layer.data)
        self.data_level: int = layer._data_level
        self.indices = indices

    def __str__(self):
        return f"location=({self.data_id}, {self.data_level}, {self.indices}) "

    def __eq__(self, other) -> bool:
        return (
            super().__eq__(other)
            and self.data_id == other.data_id
            and self.data_level == other.data_level
            and self._same_indices(other)
        )

    def _same_indices(self, other) -> bool:
        """Return True if this location has same indices as the other location.

        Returns
        -------
        bool
            True if indices are the same.
        """
        # TODO_OCTREE: Why is this sometimes ndarray and sometimes not?
        # We should normalize when the ImageLocation is constructed?
        if isinstance(self.indices, np.ndarray):
            return (self.indices == other.indices).all()
        return self.indices == other.indices

    def __hash__(self) -> int:
        """Return has of this location.

        Returns
        -------
        int
            The hash of the location.
        """
        return hash(
            (
                self.layer_ref.layer_id,
                self.data_id,
                self.data_level,
                _flatten(self.indices),
            )
        )


def _flatten(indices) -> tuple:
    """Return a flat tuple of integers to represent the indices.

    Slice objects are not hashable, so we convert them.
    """
    result = []
    for x in indices:
        if isinstance(x, slice):
            result.extend([x.start, x.stop, x.step])
        else:
            result.append(x)
    return tuple(result)

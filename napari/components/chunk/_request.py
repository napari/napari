"""ChunkRequest is passed to ChunkLoader.load_chunks().
"""
import logging
from typing import Optional, Tuple

import numpy as np

from ...layers.base.base import Layer
from ...types import ArrayLike, Dict
from ._utils import get_data_id

LOGGER = logging.getLogger("napari.async")

# We convert slices to tuple for hashing.
SliceTuple = Tuple[Optional[int], Optional[int], Optional[int]]


def _flatten(indices):
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


class ChunkKey:
    """The key for one single ChunkRequest.

    Parameters
    ----------
    layer : Layer
        The layer to load data for.
    indices : Indices
        The indices to load from the layer.

    Attributes
    ----------
    layer_id : int
        The id of the layer making the request.
    data_level : int
        The level in the data (for multi-scale).
    indices : Tuple[Optional[slice], ...]
        The indices of the slice.
    key : Tuple
        The combined key, all the identifiers together.
    """

    def __init__(self, layer: Layer, indices: Tuple[Optional[slice], ...]):
        self.layer_id = id(layer)
        self.data_id = get_data_id(layer)
        self.data_level = layer._data_level
        self.indices = indices

        combined = (
            self.layer_id,
            self.data_id,
            self.data_level,
            _flatten(self.indices),
        )
        self.key = hash(combined)

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
    key : ChunkKey
        The key of the request.
    chunks : Dict[str, ArrayLike]
        The chunk arrays we need to load.

    Attributes
    ----------
    key : ChunkKey
        The key of the request.
    chunks : Dict[str, ArrayLike]
        The chunk arrays we need to load.
    """

    def __init__(self, key: ChunkKey, chunks: Dict[str, ArrayLike]):
        # Make sure chunks dict is what we expect.
        for chunk_key, array in chunks.items():
            assert isinstance(chunk_key, str)
            assert array is not None

        self.key = key
        self.chunks = chunks

    @property
    def in_memory(self) -> bool:
        """True if all chunks are ndarrays."""
        return all(isinstance(x, np.ndarray) for x in self.chunks.values())

    def load_chunks(self):
        """Load all of our chunks now in this thread."""
        for key, array in self.chunks.items():
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
    def image(self):
        """The image chunk if we have one or None."""
        return self.chunks.get('image')

    @property
    def thumbnail_source(self):
        """The chunk to use as the thumbnail_source or None."""
        try:
            return self.chunks['thumbnail_source']
        except KeyError:
            # No thumbnail_source so return the image instead. For single-scale
            # we use the image as the thumbnail_source.
            return self.chunks.get('image')

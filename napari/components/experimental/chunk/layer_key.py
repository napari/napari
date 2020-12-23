"""LayerKey class.

We put this in its own file because (eventually) this should be the only
ChunkLoader file that imports layer.

Ideally ChunkLoader does not depend on layers at all. We may or may not
actually do that, but at the very least we want to keep track of where
we do depend on layer.
"""
from typing import NamedTuple, Optional, Tuple

from ....layers import Layer


def get_data_id(data) -> int:
    """Return the data_id to use for this layer.

    Parameters
    ----------
    layer
        The layer to get the data_id from.

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


class LayerKey(NamedTuple):
    """The key for a layer and its important properties.

    Attributes
    ----------
    layer_id : int
        The id of the layer making the request.
    data_id : int
        The id of the data in the layer.
    data_level : int
        The level in the data (for multi-scale).
    indices : Tuple[Optional[slice], ...]
        The indices of the slice.
    """

    layer_id: int
    data_id: int
    data_level: int
    indices: Tuple[Optional[slice], ...]

    def _get_hash_values(self):
        return (
            self.layer_id,
            self.data_id,
            self.data_level,
            _flatten(self.indices),
        )

    @classmethod
    def from_layer(cls, layer: Layer, indices):
        """Return LayerKey based on this layer and its indices.

        Parameters
        ----------
        layer : Layer
            Create a LayerKey for this layer.
        indices : ???
            The indices we are viewing.
        """
        return cls(
            id(layer), get_data_id(layer.data), layer._data_level, indices
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

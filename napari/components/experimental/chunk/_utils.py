"""ChunkLoader utilities.
"""
import weakref
from typing import NamedTuple, Optional

import dask.array as da
import numpy as np

from .layer_key import LayerKey


def _get_type_str(data) -> str:
    """Get human readable name for the data's type.

    Returns
    -------
    str
        A string like "ndarray" or "dask".
    """
    data_type = type(data)

    if data_type == list:
        if len(data) == 0:
            return "EMPTY"
        # Recursively get the type string of the zeroth level.
        return _get_type_str(data[0])

    if data_type == da.Array:
        # Special case this because otherwise data_type.__name__
        # below would just return "Array".
        return "dask"

    # For class numpy.ndarray this returns "ndarray"
    return data_type.__name__


class LayerRef(NamedTuple):
    """Weak reference to a layer.

    We do not want to hold a reference in case the later is deleted while
    a load is in progress. We want to let the layer get deleted, and
    when the load finishes we'll just throw it away. Since its layer is done.

    Attributes
    ----------
    layer_id : int
        The id of the layer.
    reference : weakref.ReferenceType
        A weak reference to the layer
    data_type : str
        String describing the data type the layer holds.
    """

    layer_key: LayerKey
    weak_ref: weakref.ReferenceType
    data_type: str

    @classmethod
    def create_from_layer(cls, layer, indices):
        """Return a LayerRef created from this layer.

        Parameters
        ----------
        layer
            Create the LayerRef from this layer.
        """
        layer_key = LayerKey.from_layer(layer, indices)
        return cls(layer_key, weakref.ref(layer), _get_type_str(layer.data))


class StatWindow:
    """Average value over a rolling window.

    Notes
    -----
    Inserting values once the window is full is O(1). However calculating
    the average is O(N) although using numpy.

    Parameters
    ----------
    size : int
        The size of the window.

    Attributes
    ----------
    values : ndarray
        The values in our window.
    """

    def __init__(self, size: int):
        self.size = size
        self.values = np.array([])  # float64 array

        # Once the window is full we insert values at this index, the
        # index loops through the slots circularly, forever.
        self.index = 0

    def add(self, value: float) -> None:
        """Add one value to the window.

        Parameters
        ----------
        value : float
            Add this value to the window.
        """
        if len(self.values) < self.size:
            # Not super efficient but once window is full we are O(1).
            self.values = np.append(self.values, value)
        else:
            # Window is full, poke values in circularly.
            self.values[self.index] = value
            self.index = (self.index + 1) % self.size

    @property
    def average(self) -> Optional[float]:
        """Return the average of all the values in the window.

        Returns
        -------
        float
            The average of all values in the window.
        """
        if len(self.values) == 0:
            return None
        return float(np.average(self.values))

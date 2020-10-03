"""ChunkLoader utilities.
"""
from typing import Optional

import numpy as np


def get_data_id(layer) -> int:
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
    data = layer.data
    if isinstance(data, list):
        assert data  # data should not be empty for image layers.
        return id(data[0])  # Just use the ID from the 0'th layer.

    return id(data)  # Not a list, just use it.


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

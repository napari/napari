"""ChunkLoader utilities.
"""
from typing import Optional

import dask.array as da
import numpy as np


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

    def __init__(self, size: int) -> None:
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

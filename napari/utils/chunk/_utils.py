"""ChunkLoader related utilities.
"""
import ctypes

import numpy as np


def get_data_id(layer) -> int:
    """Return the data_id to use for this layer.

    Parameters
    ----------
    layer
        The layer to get the data_id from.

    Notes
    -----
    We use data_id rather than layer_id in case someone changes the data
    out from under a layer.
    """
    data = layer.data
    if isinstance(data, list):
        # Assert for now, but shapes layers do have an empty list.
        assert data

        # Just use the ID from the 0'th layer.
        return id(data[0])

    # Not a list so just use its id.
    return id(data)


def hold_gil(seconds: float):
    """Hold the GIL for some number of seconds.

    This is used for debugging and performance testing only.
    """
    usec = seconds * 1000000
    _libc_name = ctypes.util.find_library("c")
    if _libc_name is None:
        raise RuntimeError("Cannot find libc")
    libc_py = ctypes.PyDLL(_libc_name)
    libc_py.usleep(usec)


class StatWindow:
    """Maintain an average value over some rolling window.

    Adding values is very efficient (once the window is full) but
    calculating the average is O(size), although using numpy.

    Parameters
    ----------
    size : int
        The size of the window.
    """

    def __init__(self, size: int):
        self.size = size
        self.values = np.array([])  # float64 array
        self.index = 0  # insert values here once full

    def add(self, value: float):
        """Add one value to the window.

        Parameters
        ----------
        value : float
            Add this value to the window.
        """
        if len(self.values) < self.size:
            # This isn't super efficient but we're optimizing for the case
            # when the array is full and we are just poking in values.
            self.values = np.append(self.values, value)
        else:
            # Array is full, rotate through poking in one value at a time,
            # this should be very fast.
            self.values[self.index] = value
            self.index = (self.index + 1) % self.size

    @property
    def average(self):
        """Return the average of all values in the window."""
        if len(self.values) == 0:
            return 0  # Just say zero, really there is no value.
        return np.average(self.values)

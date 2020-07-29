"""ChunkLoader related utilities.
"""
import ctypes


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

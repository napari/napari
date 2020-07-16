"""Chunk related utilities.
"""
import ctypes


def _hold_gil(seconds: float):
    """Hold the GIL for some number of seconds.

    This is used for debugging and performance testing only.
    """
    usec = seconds * 1000000
    _libc_name = ctypes.util.find_library("c")
    if _libc_name is None:
        raise RuntimeError("Cannot find libc")
    libc_py = ctypes.PyDLL(_libc_name)
    libc_py.usleep(usec)

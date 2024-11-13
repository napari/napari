"""
This module is used to prevent a known issue with numpy<2 on macOS arm64
architecture installed from pypi wheels
(https://github.com/numpy/numpy/issues/21799).

We use a method to set thread limits based on the threadpoolctl package, but
reimplemented locally to prevent adding the dependency to napari.

Note: if any issues surface with the method below, we could fall back on
depending on threadpoolctl directly.

TODO: This module can be removed once the minimum numpy version is 2+.
"""

import ctypes
import logging
import os
import platform
from pathlib import Path

import numpy as np
from packaging.version import parse as parse_version

# if run with numpy<2 on macOS arm64 architecture compiled from pypi wheels,
# then it will crash with bus error if numpy is used in different thread
# Issue reported https://github.com/numpy/numpy/issues/21799
if (
    parse_version(np.__version__) < parse_version('2')
    and platform.system() == 'Darwin'
    and platform.machine() == 'arm64'
):  # pragma: no cover
    try:
        NUMPY_VERSION_IS_THREADSAFE = (
            'cibw-run'
            not in np.show_config('dicts')['Python Information']['path']  # type: ignore
        )
    except (KeyError, TypeError):
        NUMPY_VERSION_IS_THREADSAFE = True
else:
    NUMPY_VERSION_IS_THREADSAFE = True


def limit_numpy1x_threads_on_macos_arm() -> (
    None
):  # pragma: no cover (macos only code)
    """Set openblas to use single thread on macOS arm64 to prevent numpy crash.

    On NumPy version<2 wheels on macOS ARM64 architectures, a BusError is
    raised, crashing Python, if NumPy is accessed from multiple threads.
    (See https://github.com/numpy/numpy/issues/21799.) This function uses the
    global check above (NUMPY_VERSION_IS_THREADSAFE), and, if False, it loads
    the linked OpenBLAS library and sets the number of threads to 1. This has
    performance implications but prevents nasty crashes, and anyway can be
    avoided by using more recent versions of NumPy.

    This function is loading openblas library from numpy and set number of threads to 1.
    See also:
        https://github.com/OpenMathLib/OpenBLAS/wiki/faq#how-can-i-use-openblas-in-multi-threaded-applications

    These changes seem to be sufficient to prevent the crashes.
    """
    if NUMPY_VERSION_IS_THREADSAFE:
        return
    # find openblas library
    numpy_dir = Path(np.__file__).parent
    if not (numpy_dir / '.dylibs').exists():
        logging.warning(
            'numpy .dylibs directory not found during try to prevent numpy crash'
        )
    # Recent numpy versions are built with cibuildwheel.
    # Internally, it uses delocate, which stores the openblas
    # library in the .dylibs directory.
    # Since we only patch numpy<2, we can just search for the libopenblas
    # dynamic library at this location.
    blas_lib = list((numpy_dir / '.dylibs').glob('libopenblas*.dylib'))
    if not blas_lib:
        logging.warning(
            'libopenblas not found during try to prevent numpy crash'
        )
        return
    blas = ctypes.CDLL(str(blas_lib[0]), mode=os.RTLD_NOLOAD)
    for suffix in ('', '64_', '_64'):
        openblas_set_num_threads = getattr(
            blas, f'openblas_set_num_threads{suffix}', None
        )
        if openblas_set_num_threads is not None:
            openblas_set_num_threads(1)
            break
    else:
        logging.warning('openblas_set_num_threads not found')

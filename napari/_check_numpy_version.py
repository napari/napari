"""
This module is used to prevent a known issue with numpy<2 on macOS arm64
architecture installed from pypi wheels.
Setting thread limits is inspired on threadpoolctl package to prevent adding
the dependency to napari.
However, if there is some problem in future it should be ok to
add threadpoolctl as a dependency to napari and use it directly.
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
        PROBLEMATIC_NUMPY_MACOS = (
            'cibw-run' in np.show_config('dicts')['Python Information']['path']  # type: ignore
        )
    except (KeyError, TypeError):
        PROBLEMATIC_NUMPY_MACOS = False
else:
    PROBLEMATIC_NUMPY_MACOS = False


def prevent_numpy_arm_problem() -> None:
    """Set openblas to use only one thread to prevent numpy crash on macOS arm64

    This function is loading openblas library from numpy and set number of threads to 1.
    See: https://github.com/OpenMathLib/OpenBLAS/wiki/faq#how-can-i-use-openblas-in-multi-threaded-applications
    We observe that it is enough to prevent numpy crash on macOS arm64 architecture.
    """
    if not PROBLEMATIC_NUMPY_MACOS:
        return
    # find openblas library
    numpy_dir = Path(np.__file__).parent
    if not (numpy_dir / '.dylibs').exists():
        logging.warning(
            'numpy .dylibs directory not found during try to prevent numpy crash'
        )
    # As I have checked that recent numpy versions are build using cibuildwheel.
    # It internally is using delocate, that stores openblas library in .dylibs directory.
    # As we only patch numpy<2, we can assume that it is enough to search for libopenblas dynamic library
    blas_lib = list((numpy_dir / '.dylibs').glob('libopenblas*.dylib'))
    if not blas_lib:
        logging.warning(
            'libopenblas not found during try to prevent numpy crash'
        )
        return
    blas = ctypes.CDLL(str(blas_lib[0]), mode=os.RTLD_NOLOAD)
    for suffix in ('', '64_', '_64'):
        if (
            openblas_set_num_threads := getattr(
                blas, f'openblas_set_num_threads{suffix}', None
            )
        ) is not None:
            break
    else:
        logging.warning(
            'openblas_set_num_threads not found during try to prevent numpy crash'
        )
        return
    openblas_set_num_threads(1)

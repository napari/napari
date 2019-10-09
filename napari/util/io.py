import os

import numpy as np
from skimage import io

dask_available = True
try:
    from dask import array as da
except ImportError:
    dask_available = False


def imread(filenames, *, use_dask=None, stack=True):
    """Read image files and return an array.

    If multiple images are selected, they are stacked along the 0th axis.

    Parameters
    -------
    filenames : list
        List of filenames to be opened
    use_dask : bool
        Whether to use dask to create a lazy array, rather than NumPy.
        Default is None, which is interpreted as "use if available". If set
        to True and dask is not installed, this function
    stack : bool
        Whether to stack the images in multiple files into a single array. If
        False, a list of arrays will be returned.

    Returns
    -------
    image : array
        Array of images
    """
    if dask_available and use_dask is None:
        use_dask = True
    if not dask_available and use_dask:
        raise ValueError('Dask array requested but dask is not installed.')
    images = [io.imread(filename) for filename in filenames]
    if len(images) == 1:
        image = images[0]
    else:
        if use_dask:
            image = da.stack(images)
        else:
            image = np.stack(images)

    return image


def magic_read(filenames, *, use_dask=None, stack=True):
    """Dispatch the appropriate reader given some files.

    The files are assumed to all have the same type.

    Parameters
    -------
    filenames : list
        List of filenames to be opened
    use_dask : bool
        Whether to use dask to create a lazy array, rather than NumPy.
        Default is None, which is interpreted as "use if available". If set
        to True and dask is not installed, this function
    stack : bool
        Whether to stack the images in multiple files into a single array. If
        False, a list of arrays will be returned.

    Returns
    -------
    image : array-like
        Array or list of images
    """
    if len(filenames) == 0:
        return None
    ext = os.path.splitext(filenames[0])[-1]
    if ext == '.zarr':
        if not dask_available:
            raise ValueError('Dask is required to open zarr files.')
        if len(filenames) == 1:
            return da.from_zarr(filenames[0])
        else:
            loaded = [da.from_zarr(f) for f in filenames]
            if stack:
                return da.stack(loaded)
            else:
                return loaded
    else:  # assume proper image extension
        return imread(filenames, use_dask=use_dask, stack=stack)

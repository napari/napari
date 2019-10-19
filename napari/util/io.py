import os
from glob import glob

import numpy as np
from skimage import io
from skimage.io.collection import alphanumeric_key

from dask import delayed
from dask import array as da


def magic_read(filenames, *, use_dask=None, stack=True):
    """Dispatch the appropriate reader given some files.

    The files are assumed to all have the same shape.

    Parameters
    -------
    filenames : list
        List of filenames or directories to be opened
    use_dask : bool
        Whether to use dask to create a lazy array, rather than NumPy.
        Default of None will resolve to True if filenames contains more than
        one image, False otherwise.
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
    if isinstance(filenames, str):
        filenames = [filenames]  # ensure list

    # replace folders with their contents
    filenames_expanded = []
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        # zarr files are folders, but should be read as 1 file
        if os.path.isdir(filename) and not ext == '.zarr':
            dir_contents = sorted(
                glob(os.path.join(filename, '*.*')), key=alphanumeric_key
            )
            # remove subdirectories
            dir_contents_files = filter(
                lambda f: not os.path.isdir(f), dir_contents
            )
            filenames_expanded.extend(dir_contents_files)
        else:
            filenames_expanded.append(filename)

    if use_dask is None:
        use_dask = len(filenames_expanded) > 1

    # then, read in images
    images = []
    shape = None
    for filename in filenames_expanded:
        ext = os.path.splitext(filename)[-1]
        if ext == '.zarr':
            image = da.from_zarr(filename)
            if shape is None:
                shape = image.shape
        else:
            if shape is None:
                image = io.imread(filename)
                shape = image.shape
                dtype = image.dtype
            if use_dask:
                image = da.from_delayed(
                    delayed(io.imread)(filename), shape=shape, dtype=dtype
                )
            elif len(images) > 0:  # not read by shape clause
                image = io.imread(filename)
        images.append(image)
    if len(images) == 1:
        image = images[0]
    else:
        if stack:
            if use_dask:
                image = da.stack(images)
            else:
                image = np.stack(images)
        else:
            image = images  # return a list
    return image

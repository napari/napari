import os
import re

from glob import glob
from pathlib import Path

import numpy as np

from dask import delayed
from dask import array as da


def imsave(filename: str, data: np.ndarray):
    """custom imaplementation of imread to avoid skimage dependecy"""
    ext = os.path.splitext(filename)[1]
    if ext in [".tif", "tiff"]:
        import tifffile

        tifffile.imsave(filename, data)
    else:
        import imageio

        imageio.imsave(filename, data)


def imread(filename: str):
    """custom imaplementation of imread to avoid skimage dependecy"""
    ext = os.path.splitext(filename)[1]
    if ext in [".tif", "tiff", ".lsm"]:
        import tifffile

        image = tifffile.imread(filename)
    else:
        import imageio

        image = imageio.imread(filename)

    if not hasattr(image, 'ndim'):
        return image

    if image.ndim > 2:
        if image.shape[-1] not in (3, 4) and image.shape[-3] in (3, 4):
            image = np.swapaxes(image, -1, -3)
            image = np.swapaxes(image, -2, -3)
    return image


if not os.environ.get("NO_SKIMAGE", False):
    try:
        from skimage.io import imread as _imread
    except ImportError:
        _imread = imread
else:
    _imread = imread


def _alphanumeric_key(s):
    """Convert string to list of strings and ints that gives intuitive sorting.

    Parameters
    ----------
    s : string

    Returns
    -------
    k : a list of strings and ints

    Examples
    --------
    >>> _alphanumeric_key('z23a')
    ['z', 23, 'a']
    >>> filenames = ['f9.10.png', 'e10.png', 'f9.9.png', 'f10.10.png',
    ...              'f10.9.png']
    >>> sorted(filenames)
    ['e10.png', 'f10.10.png', 'f10.9.png', 'f9.10.png', 'f9.9.png']
    >>> sorted(filenames, key=_alphanumeric_key)
    ['e10.png', 'f9.9.png', 'f9.10.png', 'f10.9.png', 'f10.10.png']
    """
    k = [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
    return k


def magic_imread(filenames, *, use_dask=None, stack=True):
    """Dispatch the appropriate reader given some files.

    The files are assumed to all have the same shape.

    Parameters
    -------
    filenames : list
        List of filenames or directories to be opened.
        A list of `pathlib.Path` objects and a single filename or `Path` object
        are also accepted.
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
    # cast Path to string
    if isinstance(filenames, Path):
        filenames = filenames.as_posix()

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
                glob(os.path.join(filename, '*.*')), key=_alphanumeric_key
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
            image, zarr_shape = read_zarr_dataset(filename)
            if shape is None:
                shape = zarr_shape
        else:
            if shape is None:
                image = _imread(filename)
                shape = image.shape
                dtype = image.dtype
            if use_dask:
                image = da.from_delayed(
                    delayed(_imread)(filename), shape=shape, dtype=dtype
                )
            elif len(images) > 0:  # not read by shape clause
                image = _imread(filename)
        images.append(image)
    if len(images) == 1:
        image = images[0]
    else:
        if stack:
            if use_dask:
                image = da.stack(images)
            else:
                try:
                    image = np.stack(images)
                except ValueError as e:
                    if 'input arrays must have the same shape' in str(e):
                        msg = (
                            'To stack multiple files into a single array with '
                            'numpy, all input arrays must have the same shape.'
                            ' Set `use_dask` to True to stack arrays with '
                            'different shapes.'
                        )
                        raise ValueError(msg) from e
        else:
            image = images  # return a list
    return image


def read_zarr_dataset(path):
    """Read a zarr dataset, including an array or a group of arrays.

    Parameters
    --------
    path : str
        Path to directory ending in '.zarr'. Path can contain either an array
        or a group of arrays in the case of pyramid data.
    Returns
    -------
    image : array-like
        Array or list of arrays
    shape : tuple
        Shape of array or first array in list
    """
    if os.path.exists(os.path.join(path, '.zarray')):
        # load zarr array
        image = da.from_zarr(path)
        shape = image.shape
    elif os.path.exists(os.path.join(path, '.zgroup')):
        # else load zarr all arrays inside file, useful for pyramid data
        image = []
        for subpath in sorted(os.listdir(path)):
            if not subpath.startswith('.'):
                image.append(read_zarr_dataset(os.path.join(path, subpath))[0])
        shape = image[0].shape
    else:
        raise ValueError(f"Not a zarr dataset or group: {path}")
    return image, shape

import os
import dask.array as da
from dask import delayed
from skimage.io import imread
from skimage.external import tifffile as tf
from glob import glob
import re

delayed_read = delayed(imread)


def is_lls_folder(path):
    """this should be a fast function that doesn't actually read any data.
    It should simply return True if it is capable of handling the path."""
    if not os.path.isdir(path):
        return False
    tiffs = 0
    for fname in os.listdir(path):
        # ignore all files but tifs
        if not fname.endswith('.tif'):
            continue
        if not 'msecAbs' in fname and 'stack' in fname:
            return False
        else:
            tiffs += 1
    return tiffs > 0


# two options here... the reader can either accept the viewer instance, and
# call viewer.add_image (or even other layers) itself,
# or it can return args and kwargs that would be passed to viewer.add_image
# the first case is implemented here
def read_lls_folder(directory, viewer):
    """Take a directory and load images in the viewer
    
    Parameters
    ----------
    directory : str
        Path to directory
    viewer : napari.Viewer
        viewer instance
    """
    i = 0
    channels = []
    waves = []
    shape = None
    dtype = None
    while True:
        ch_files = glob(os.path.join(directory, f'*ch{i}*.tif'))
        if not ch_files:
            break
        if not (shape and dtype):
            tif = tf.TiffFile(ch_files[0])
            shape = tif.series[0].shape
            dtype = tif.series[0].dtype
            dx = tif.series[0][0].x_resolution or (1, 1)
            dx = dx[1] / dx[0]
            try:
                dz = tif.series[0][0].imagej_tags.get('spacing', 1)
            except AttributeError:
                dz = 1
        channels.append(
            da.stack(
                [
                    da.from_delayed(delayed_read(fn), shape, dtype)
                    for fn in ch_files
                ]
            )
        )
        waves.append(re.search('(\d+)nm_', ch_files[0]).groups()[0])
        i += 1

    stack = da.stack(channels)
    scale = [1] * stack.ndim
    scale[-3] = dz / dx
    viewer.add_image(stack, channel_axis=0, rgb=False, scale=scale, name=waves)
    viewer.dims.ndisplay = 3


"""
For a class-free, functional API, all IO plugins declare their functionality
with two module level variables:
    READERS : a list of 2-tuples (checker, reader)
        `checker` is a function that returns True if it recognizes a
            directory as something it can handle
        `reader` is a function that accepts args (path, viewer) and
            adds layers to the viewer given a path.
    WRITERS : a list of writing functions
        not implemented

Alternatively, we could define a class-based API.
"""
READERS = [(is_lls_folder, read_lls_folder)]
WRITERS = []

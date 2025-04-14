import os
import struct
import warnings

import numpy as np

from napari._version import __version__
from napari.utils.notifications import show_warning
from napari.utils.translations import trans


def imsave(filename: str, data: 'np.ndarray'):
    """Custom implementation of imsave to avoid skimage dependency.

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    ext = os.path.splitext(filename)[1].lower()
    # If no file extension was specified, choose .png by default
    if ext == '':
        if (
            data.ndim == 2 or (data.ndim == 3 and data.shape[-1] in {3, 4})
        ) and not np.issubdtype(data.dtype, np.floating):
            ext = '.png'
        else:
            ext = '.tif'
            filename = filename + ext
    # not all file types can handle float data
    if ext not in [
        '.tif',
        '.tiff',
        '.bsdf',
        '.im',
        '.lsm',
        '.npz',
        '.stk',
    ] and np.issubdtype(data.dtype, np.floating):
        show_warning(
            trans._(
                'Image was not saved, because image data is of dtype float.\nEither convert dtype or save as different file type (e.g. TIFF).'
            )
        )
        return
    # Save screenshot image data to output file
    if ext in ['.png']:
        imsave_png(filename, data)
    elif ext in ['.tif', '.tiff']:
        imsave_tiff(filename, data)
    else:
        import imageio.v3 as iio

        iio.imwrite(filename, data)  # for all other file extensions


def imsave_png(filename, data):
    """Save .png image to file

    PNG images created in napari have a digital watermark.
    The napari version info is embedded into the bytes of the PNG.

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    import imageio.v3 as iio
    import PIL.PngImagePlugin

    # Digital watermark, adds info about the napari version to the bytes of the PNG file
    pnginfo = PIL.PngImagePlugin.PngInfo()
    pnginfo.add_text(
        'Software', f'napari version {__version__} https://napari.org/'
    )
    iio.imwrite(
        filename,
        data,
        extension='.png',
        plugin='pillow',
        pnginfo=pnginfo,
    )


def imsave_tiff(filename, data):
    """Save .tiff image to file

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    import tifffile

    if data.dtype == bool:
        tifffile.imwrite(filename, data)
    else:
        try:
            tifffile.imwrite(
                filename,
                data,
                # compression arg structure since tifffile 2022.7.28
                compression='zlib',
                compressionargs={'level': 1},
            )
        except struct.error:
            # regular tiffs don't support compressed data >4GB
            # in that case a struct.error is raised, and we write with the
            # bigtiff flag. (The flag is not on by default because it is
            # not as widely supported as normal tiffs.)
            tifffile.imwrite(
                filename,
                data,
                compression='zlib',
                compressionargs={'level': 1},
                bigtiff=True,
            )


def __getattr__(name: str):
    if name in {
        'imsave_extensions',
        'write_csv',
        'read_csv',
        'csv_to_layer_data',
        'read_zarr_dataset',
    }:
        warnings.warn(
            trans._(
                '{name} was moved from napari.utils.io in v0.4.17. Import it from napari_builtins.io instead.',
                deferred=True,
                name=name,
            ),
            FutureWarning,
            stacklevel=2,
        )
        import napari_builtins.io

        return getattr(napari_builtins.io, name)

    raise AttributeError(f'module {__name__} has no attribute {name}')

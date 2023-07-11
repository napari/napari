import os
import warnings
from typing import TYPE_CHECKING

from napari._version import __version__
from napari.utils.translations import trans

if TYPE_CHECKING:
    import numpy as np


def imsave(filename: str, data: "np.ndarray"):
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
    if ext == "":
        ext = ".png"
    # Save screenshot image data to output file
    if ext in [".png"]:
        imsave_png(filename, data)
    elif ext in [".tif", ".tiff"]:
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
        "Software", f"napari version {__version__} https://napari.org/"
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

    compression_instead_of_compress = False
    try:
        current_version = tuple(
            int(x) for x in tifffile.__version__.split('.')[:3]
        )
        compression_instead_of_compress = current_version >= (2021, 6, 6)
    except Exception:  # noqa: BLE001
        # Just in case anything goes wrong in parsing version number
        # like repackaging on linux or anything else we fallback to
        # using compress
        warnings.warn(
            trans._(
                'Error parsing tiffile version number {version_number}',
                deferred=True,
                version_number=f"{tifffile.__version__:!r}",
            )
        )

    if compression_instead_of_compress:
        # 'compression' scheme is more complex. See:
        # https://forum.image.sc/t/problem-saving-generated-labels-in-cellpose-napari/54892/8
        tifffile.imwrite(filename, data, compression=('zlib', 1))
    else:  # older version of tifffile since 2021.6.6  this is deprecated
        tifffile.imwrite(filename, data, compress=1)


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

    raise AttributeError(f"module {__name__} has no attribute {name}")

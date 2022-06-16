import os
import warnings

import numpy as np

from ..utils.translations import trans


def imsave(filename: str, data: np.ndarray):
    """Custom implementation of imsave to avoid skimage dependency.

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    ext = os.path.splitext(filename)[1]
    if ext in [".tif", ".tiff"]:
        import tifffile

        compression_instead_of_compress = False
        try:
            current_version = tuple(
                int(x) for x in tifffile.__version__.split('.')[:3]
            )
            compression_instead_of_compress = current_version >= (2021, 6, 6)
        except Exception:
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
    else:
        import imageio

        imageio.imsave(filename, data)

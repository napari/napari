import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ..types import FullLayerData
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


def imsave_extensions() -> Tuple[str, ...]:
    """Valid extensions of files that imsave can write to.

    Returns
    -------
    tuple
        Valid extensions of files that imsave can write to.
    """
    _warn_about_deprecated_import('imsave_extensions')
    import napari_builtins.io

    return napari_builtins.io.imsave_extensions()


def write_csv(
    filename: str,
    data: Union[List, np.ndarray],
    column_names: Optional[List[str]] = None,
):
    """Write a csv file.

    Parameters
    ----------
    filename : str
        Filename for saving csv.
    data : list or ndarray
        Table values, contained in a list of lists or an ndarray.
    column_names : list, optional
        List of column names for table data.
    """
    _warn_about_deprecated_import('write_csv')
    import napari_builtins.io

    return napari_builtins.io.write_csv(filename, data, column_names)


def imread(filename: str) -> np.ndarray:
    """Custom implementation of imread to avoid skimage dependency.

    Parameters
    ----------
    filename : string
        The path from which to read the image.

    Returns
    -------
    data : np.ndarray
        The image data.
    """
    _warn_about_deprecated_import('imread')
    import napari_builtins.io

    return napari_builtins.io.imread(filename)


def read_csv(
    filename: str, require_type: Optional[str] = None
) -> Tuple[np.ndarray, List[str], Optional[str]]:
    """Return CSV data only if column names match format for ``require_type``.

    Reads only the first line of the CSV at first, then optionally raises an
    exception if the column names are not consistent with a known format, as
    determined by the ``require_type`` argument and
    :func:`_guess_layer_type_from_column_names`.

    Parameters
    ----------
    filename : str
        Path of file to open
    require_type : str, optional
        The desired layer type. If provided, should be one of the keys in
        ``csv_reader_functions`` or the string "any".  If ``None``, data, will
        not impose any format requirements on the csv, and data will always be
        returned.  If ``any``, csv must be recognized as one of the valid layer
        data formats, otherwise a ``ValueError`` will be raised.  If a specific
        layer type string, then a ``ValueError`` will be raised if the column
        names are not of the predicted format.

    Returns
    -------
    (data, column_names, layer_type) : Tuple[np.array, List[str], str]
        The table data and column names from the CSV file, along with the
        detected layer type (string).

    Raises
    ------
    ValueError
        If the column names do not match the format requested by
        ``require_type``.
    """
    _warn_about_deprecated_import('read_csv')
    import napari_builtins.io

    return napari_builtins.io.read_csv(filename, require_type)


def csv_to_layer_data(
    path: str, require_type: Optional[str] = None
) -> Optional[FullLayerData]:
    """Return layer data from a CSV file if detected as a valid type.

    Parameters
    ----------
    path : str
        Path of file to open
    require_type : str, optional
        The desired layer type. If provided, should be one of the keys in
        ``csv_reader_functions`` or the string "any".  If ``None``,
        unrecognized CSV files will simply return ``None``.  If ``any``,
        unrecognized CSV files will raise a ``ValueError``.  If a specific
        layer type string, then a ``ValueError`` will be raised if the column
        names are not of the predicted format.

    Returns
    -------
    layer_data : tuple, or None
        3-tuple ``(array, dict, str)`` (points data, metadata, layer_type) if
        CSV is recognized as a valid type.

    Raises
    ------
    ValueError
        If ``require_type`` is not ``None``, but the CSV is not detected as a
        valid data format.
    """
    _warn_about_deprecated_import('csv_to_layer_data')
    import napari_builtins.io

    return napari_builtins.io.csv_to_layer_data(path, require_type)


def read_zarr_dataset(path):
    """Read a zarr dataset, including an array or a group of arrays.

    Parameters
    ----------
    path : str
        Path to directory ending in '.zarr'. Path can contain either an array
        or a group of arrays in the case of multiscale data.

    Returns
    -------
    image : array-like
        Array or list of arrays
    shape : tuple
        Shape of array or first array in list
    """
    _warn_about_deprecated_import('read_zarr_dataset')
    import napari_builtins.io

    return napari_builtins.io.read_zarr_dataset(path)


def _warn_about_deprecated_import(name):
    warnings.warn(
        trans._(
            '{name} was moved from napari.utils.io in 0.4.17. Import it from napari_builtins.io instead.',
            deferred=True,
            name=name,
        ),
        DeprecationWarning,
        stacklevel=3,
    )

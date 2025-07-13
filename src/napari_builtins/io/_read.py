import csv
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import imageio.v3 as iio
import numpy as np
from dask import delayed

from napari.utils.misc import abspath_or_url
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.types import FullLayerData, LayerData, ReaderFunction

PathOrStr = Union[str, Path]


def _alphanumeric_key(s: str) -> list[Union[str, int]]:
    """Convert string to list of strings and ints that gives intuitive sorting."""
    return [int(c) if c.isdigit() else c for c in re.split(r'([0-9]+)', s)]


URL_REGEX = re.compile(r'https?://|ftps?://|file://|file:\\')


def _is_url(filename: Union[str, Path]) -> bool:
    """Return True if string is an http or ftp path."""
    return isinstance(filename, str) and URL_REGEX.match(filename) is not None


def imread(filename: Union[str, Path]) -> np.ndarray:
    """Dispatch reading images to imageio.v3 imread."""
    filename_str = str(filename)
    filename_str = abspath_or_url(filename_str)

    # Skip .npy handling for URLs
    if _is_url(filename_str):
        return iio.imread(filename_str)

    ext = os.path.splitext(filename_str)[1].lower()
    if ext == '.npy':
        return np.load(filename_str)

    return iio.imread(filename_str)


def _guess_zarr_path(path: Union[str, Path]) -> bool:
    """Guess whether string path is part of a zarr hierarchy."""
    return any(part.endswith('.zarr') for part in Path(path).parts)


def read_zarr_dataset(path: Union[str, Path]) -> tuple[Any, tuple[int, ...]]:
    """Read a zarr dataset, including an array or a group of arrays."""
    path_obj = Path(path)
    if (path_obj / '.zarray').exists():
        image = da.from_zarr(path_obj)
        return image, image.shape
    if (path_obj / '.zgroup').exists():
        subpaths = [
            subpath
            for subpath in sorted(path_obj.iterdir())
            if not subpath.name.startswith('.') and subpath.is_dir()
        ]
        images = [read_zarr_dataset(subpath)[0] for subpath in subpaths]
        if not images:
            raise ValueError(
                trans._('No arrays found in zarr group', deferred=True)
            )
        return images, images[0].shape
    if (path_obj / 'zarr.json').exists():
        import zarr

        store = zarr.open(path_obj)
        if isinstance(store, zarr.Array):
            image = da.from_zarr(store)
            return image, image.shape
        images = [da.from_zarr(store[k]) for k in sorted(store)]
        if not images:
            raise ValueError(
                trans._('No arrays found in zarr group', deferred=True)
            )
        return images, images[0].shape
    raise ValueError(
        trans._(
            'Not a zarr dataset or group: {path}', deferred=True, path=path
        )
    )


def magic_imread(
    filenames: Union[PathOrStr, list[PathOrStr]],
    *,
    use_dask: Optional[bool] = None,
    stack: bool = True,
) -> Optional[Union[da.Array, np.ndarray, list]]:
    """Dispatch the appropriate reader given some files."""
    # Normalize input to list of strings
    if isinstance(filenames, (list, tuple)):
        _filenames = [str(f) for f in filenames]
    else:
        _filenames = [str(filenames)]

    if not _filenames:
        raise ValueError(trans._('No files found', deferred=True))

    # Expand directories
    filenames_expanded: list[str] = []
    for filename in _filenames:
        if (
            os.path.isdir(filename)
            and not _guess_zarr_path(filename)
            and not _is_url(filename)
        ):
            # List non-hidden files in directory
            dir_files = [
                os.path.join(filename, f)
                for f in os.listdir(filename)
                if not f.startswith('.')
                and os.path.isfile(os.path.join(filename, f))
            ]
            dir_files.sort(key=_alphanumeric_key)
            filenames_expanded.extend(dir_files)
        else:
            filenames_expanded.append(filename)

    if not filenames_expanded:
        raise ValueError(
            trans._(
                'No valid files found in: {filenames}',
                deferred=True,
                filenames=filenames,
            )
        )

    # Determine if we should use dask
    use_dask = len(filenames_expanded) > 1 if use_dask is None else use_dask

    images = []
    shape, dtype = None, None

    for filename in filenames_expanded:
        if _guess_zarr_path(filename):
            try:
                image, img_shape = read_zarr_dataset(filename)
                if len(img_shape) == 1:  # Skip 1D images
                    continue
                images.append(image)
                if shape is None:
                    shape = img_shape
            except Exception:
                if len(filenames_expanded) == 1:
                    raise
                continue  # Skip problematic files in batch mode
        else:
            if use_dask and shape is not None:
                # Create delayed reads for subsequent files
                image = da.from_delayed(
                    delayed(imread)(filename), shape=shape, dtype=dtype
                )
            else:
                # Read first file immediately
                image = imread(filename)
                if shape is None:
                    shape = image.shape
                    dtype = image.dtype
            images.append(image)

    if not images:
        return None

    if len(images) == 1:
        return images[0]

    if stack:
        try:
            return da.stack(images) if use_dask else np.stack(images)
        except ValueError as e:
            if 'input arrays must have the same shape' in str(e):
                raise ValueError(
                    trans._(
                        'To stack multiple files, all images must have the same shape. '
                        'Set `use_dask=True` to support different shapes.',
                        deferred=True,
                    )
                ) from e
            raise
    return images


def _points_csv_to_layerdata(
    table: np.ndarray, column_names: list[str]
) -> 'FullLayerData':
    """Convert CSV data to Points LayerData."""
    data_axes = [
        i for i, cn in enumerate(column_names) if cn.startswith('axis-')
    ]
    if not data_axes:
        raise ValueError(trans._('No coordinate columns found', deferred=True))

    data = table[:, data_axes].astype(float)

    # Handle properties
    prop_cols = [
        i
        for i, cn in enumerate(column_names)
        if i not in data_axes and (i != 0 or cn != 'index')
    ]
    meta: dict = {}
    if prop_cols:
        meta['properties'] = {
            column_names[i]: table[:, i].astype(float)  # Try float first
            for i in prop_cols
        }
    return data, meta, 'points'


def _shapes_csv_to_layerdata(
    table: np.ndarray, column_names: list[str]
) -> 'FullLayerData':
    """Convert CSV data to Shapes LayerData."""
    try:
        index_col = column_names.index('index')
        type_col = column_names.index('shape-type')
        vertex_col = column_names.index('vertex-index')
    except ValueError:
        raise ValueError(
            trans._('CSV missing required columns for shapes', deferred=True)
        )

    data_axes = [
        i for i, cn in enumerate(column_names) if cn.startswith('axis-')
    ]
    if len(data_axes) < 2:
        raise ValueError(
            trans._(
                'Insufficient coordinate columns for shapes', deferred=True
            )
        )

    raw_data = table[:, data_axes].astype(float)
    indices = table[:, index_col].astype(int)

    # Group vertices by shape index
    transitions = np.where(np.diff(indices))[0] + 1
    boundaries = [0] + transitions.tolist() + [len(table)]

    if len(boundaries) - 1 != indices[-1] + 1:
        raise ValueError(trans._('Shape index mismatch', deferred=True))

    data, shape_types = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        shape_types.append(table[start, type_col])
        data.append(raw_data[start:end])

    return data, {'shape_type': shape_types}, 'shapes'


def _guess_layer_type_from_column_names(
    column_names: list[str],
) -> Optional[str]:
    """Guess layer type from CSV column names."""
    required_shapes = {'index', 'shape-type', 'vertex-index'}
    if required_shapes.issubset(column_names) and any(
        c.startswith('axis-') for c in column_names
    ):
        return 'shapes'
    if any(c.startswith('axis-') for c in column_names):
        return 'points'
    return None


def read_csv(
    filename: str, require_type: Optional[str] = None
) -> tuple[np.ndarray, list[str], Optional[str]]:
    """Read CSV file with format validation."""
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        column_names = next(reader)
        layer_type = _guess_layer_type_from_column_names(column_names)

        if require_type:
            if not layer_type:
                raise ValueError(
                    trans._(
                        'File not recognized as valid layer data: {filename}',
                        deferred=True,
                        filename=filename,
                    )
                )
            if require_type != 'any' and layer_type != require_type:
                raise ValueError(
                    trans._(
                        'Expected {expected} data, found {actual} in {filename}',
                        deferred=True,
                        expected=require_type,
                        actual=layer_type,
                        filename=filename,
                    )
                )

        data = np.array(list(reader))
    return data, column_names, layer_type


csv_reader_functions = {
    'points': _points_csv_to_layerdata,
    'shapes': _shapes_csv_to_layerdata,
}


def csv_to_layer_data(
    path: str, require_type: Optional[str] = None
) -> Optional['FullLayerData']:
    """Convert CSV file to layer data."""
    try:
        table, col_names, layer_type = read_csv(path, require_type or 'any')
        if layer_type and layer_type in csv_reader_functions:
            return csv_reader_functions[layer_type](table, col_names)
    except Exception:
        if not require_type:
            return None
        raise
    return None


def _csv_reader(path: Union[str, Sequence[str]]) -> list['LayerData']:
    paths = [path] if isinstance(path, str) else path
    return [
        layer_data
        for p in paths
        if (layer_data := csv_to_layer_data(p, require_type=None))
    ]


def _magic_imreader(path: Union[str, list[str]]) -> list['LayerData']:
    return [(magic_imread(path),)]


def napari_get_reader(
    path: Union[str, list[str]],
) -> Optional['ReaderFunction']:
    """Main reader dispatch function."""
    # Handle CSV files
    if isinstance(path, str) and path.endswith('.csv'):
        return _csv_reader
    if isinstance(path, list) and all(p.endswith('.csv') for p in path):
        return _csv_reader

    # Handle image/zarr files
    return _magic_imreader

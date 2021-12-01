"""
Internal napari hook implementations to be registered by the plugin manager
"""
import os
import shutil
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from napari_plugin_engine import napari_hook_implementation

from ..types import (
    FullLayerData,
    LayerData,
    ReaderFunction,
    WriterFunction,
    image_reader_to_layerdata_reader,
)
from ..utils.io import (
    READER_EXTENSIONS,
    csv_to_layer_data,
    imsave,
    imsave_extensions,
    magic_imread,
    write_csv,
)
from ..utils.misc import abspath_or_url


def csv_reader_function(path: Union[str, Sequence[str]]) -> List[LayerData]:
    if not isinstance(path, str):
        out: List[LayerData] = []
        for p in path:
            layer_data = csv_to_layer_data(p, require_type=None)
            if layer_data:
                out.append(layer_data)
        return out
    else:
        layer_data = csv_to_layer_data(path, require_type=None)
        return [layer_data] if layer_data else []


def npy_to_layer_data(path: Union[str, Sequence[str]]) -> List[LayerData]:
    if isinstance(path, str):
        return [(np.load(path),)]

    return [(np.load(p),) for p in path]


@napari_hook_implementation(trylast=True)
def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]:
    """Our internal fallback file reader at the end of the reader plugin chain.

    This will assume that the filepath is an image, and will pass all of the
    necessary information to viewer.add_image().

    Parameters
    ----------
    path : str
        path to file/directory

    Returns
    -------
    callable
        function that returns layer_data to be handed to viewer._add_layer_data
    """
    if isinstance(path, str):
        if path.endswith('.csv'):
            return csv_reader_function
        if os.path.isdir(path):
            return image_reader_to_layerdata_reader(magic_imread)
        if path.endswith('.npy'):
            return npy_to_layer_data
        path = [path]

    if all(str(x).lower().endswith(tuple(READER_EXTENSIONS)) for x in path):
        return image_reader_to_layerdata_reader(magic_imread)

    return None


@napari_hook_implementation(trylast=True)
def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback image writer at the end of the plugin chain.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If meta['rgb'] is ``True`` then the
        data should be interpreted as RGB or RGBA. If ``meta['multiscale']`` is
        ``True``, then the data should be interpreted as a multiscale image.
    meta : dict
        Image metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        path += '.tif'
        ext = '.tif'

    if ext in imsave_extensions():
        imsave(path, data)
        return path

    return None


@napari_hook_implementation(trylast=True)
def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback labels writer at the end of the plugin chain.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If meta['rgb'] is ``True`` then the
        data should be interpreted as RGB or RGBA. If ``meta['multiscale']`` is
        ``True``, then the data should be interpreted as a multiscale image.
    meta : dict
        Image metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    dtype = data.dtype if data.dtype.itemsize >= 4 else np.uint32
    return napari_write_image(path, np.asarray(data, dtype=dtype), meta)


@napari_hook_implementation(trylast=True)
def napari_write_points(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback points writer at the end of the plugin chain.

    Append ``.csv`` extension to the filename if it is not already there.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array (N, D)
        Coordinates for N points in D dimensions.
    meta : dict
        Points metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    ext = os.path.splitext(path)[1]
    if ext == '':
        path += '.csv'
    elif ext != '.csv':
        # If an extension is provided then it must be `.csv`
        return None

    properties = meta.get('properties', {})
    # TODO: we need to change this to the axis names once we get access to them
    # construct table from data
    column_names = ['axis-' + str(n) for n in range(data.shape[1])]
    if properties:
        column_names += properties.keys()
        prop_table = [
            np.expand_dims(col, axis=1) for col in properties.values()
        ]
    else:
        prop_table = []

    # add index of each point
    column_names = ['index'] + column_names
    indices = np.expand_dims(list(range(data.shape[0])), axis=1)
    table = np.concatenate([indices, data] + prop_table, axis=1)

    # write table to csv file
    write_csv(path, table, column_names)
    return path


@napari_hook_implementation(trylast=True)
def napari_write_shapes(path: str, data: Any, meta: dict) -> Optional[str]:
    """Our internal fallback points writer at the end of the plugin chain.

    Append ``.csv`` extension to the filename if it is not already there.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : list of array (N, D)
        List of coordinates for shapes, each with for N vertices in D
        dimensions.
    meta : dict
        Points metadata.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """
    ext = os.path.splitext(path)[1]
    if ext == '':
        path += '.csv'
    elif ext != '.csv':
        # If an extension is provided then it must be `.csv`
        return None

    shape_type = meta.get('shape_type', ['rectangle'] * len(data))
    # No data passed so nothing written
    if len(data) == 0:
        return None

    # TODO: we need to change this to the axis names once we get access to them
    # construct table from data
    n_dimensions = max(s.shape[1] for s in data)
    column_names = ['axis-' + str(n) for n in range(n_dimensions)]

    # add shape id and vertex id of each vertex
    column_names = ['index', 'shape-type', 'vertex-index'] + column_names

    # concatenate shape data into 2D array
    len_shapes = [s.shape[0] for s in data]
    all_data = np.concatenate(data)
    all_idx = np.expand_dims(
        np.concatenate([np.repeat(i, s) for i, s in enumerate(len_shapes)]),
        axis=1,
    )
    all_types = np.expand_dims(
        np.concatenate(
            [np.repeat(shape_type[i], s) for i, s in enumerate(len_shapes)]
        ),
        axis=1,
    )
    all_vert_idx = np.expand_dims(
        np.concatenate([np.arange(s) for s in len_shapes]), axis=1
    )

    table = np.concatenate(
        [all_idx, all_types, all_vert_idx, all_data], axis=1
    )

    # write table to csv file
    write_csv(path, table, column_names)
    return path


@napari_hook_implementation(trylast=True)
def napari_get_writer(
    path: str, layer_types: List[str]
) -> Optional[WriterFunction]:
    """Our internal fallback file writer at the end of the writer plugin chain.

    This will create a new folder from the path and call
    ``napari_write_<layer>`` for each layer using the ``layer.name`` variable
    to modify the path such that the layers are written to unique files in the
    folder. It will use the default builtin writer for each layer type.

    Parameters
    ----------
    path : str
        path to file/directory

    Returns
    -------
    callable
        function that accepts the path and a list of layer_data (where
        layer_data is ``(data, meta, layer_type)``) and writes each layer.
    """
    # normally, a plugin would do some logic here to decide whether it supports
    # the ``path`` extension and layer_types.  But because this is our builtin
    # "last resort" implementation, we just immediately hand back the writer
    # function, and let it throw an exception if it fails.
    return write_layer_data_with_plugins


def write_layer_data_with_plugins(
    path: str,
    layer_data: List[FullLayerData],
    *,
    plugin_name: Optional[str] = 'builtins',
) -> List[str]:
    """Write layer data out into a folder one layer at a time.

    Call ``napari_write_<layer>`` for each layer using the ``layer.name``
    variable to modify the path such that the layers are written to unique
    files in the folder.

    If ``plugin_name`` is ``None`` then we just directly call
    ``plugin_manager.hook.napari_write_<layer>()`` which will loop through
    implementations and stop when the first one returns a non-None result. The
    order in which implementations are called can be changed with the
    implementation sorter/disabler.

    If ``plugin_name`` is provided, then we call the
    ``napari_write_<layer_type>`` for that plugin, and if it fails we error.
    By default, we restrict this function to using only napari ``builtins``
    plugins.

    Parameters
    ----------
    path : str
        path to file/directory
    layer_data : list of napari.types.LayerData
        List of layer_data, where layer_data is ``(data, meta, layer_type)``.
    plugin_name : str, optional
        Name of the plugin to use for saving. If None then all plugins
        corresponding to appropriate hook specification will be looped
        through to find the first one that can save the data. By default,
        only builtin napari implementations are used.

    Returns
    -------
    list of str
        A list of any filepaths that were written.
    """
    from tempfile import TemporaryDirectory

    from . import plugin_manager

    # remember whether it was there to begin with
    already_existed = os.path.exists(path)
    # Try and make directory based on current path if it doesn't exist
    if not already_existed:
        os.makedirs(path)

    written: List[str] = []  # the files that were actually written
    try:
        # build in a temporary directory and then move afterwards,
        # it makes cleanup easier if an exception is raised inside.
        with TemporaryDirectory(dir=path) as tmp:
            # Loop through data for each layer
            for layer_data_tuple in layer_data:
                data, meta, layer_type = layer_data_tuple
                # Get hook caller according to layer type
                hook_caller = getattr(
                    plugin_manager.hook, f'napari_write_{layer_type}'
                )
                # Create full path using name of layer
                full_path = abspath_or_url(os.path.join(tmp, meta['name']))
                # Write out data using first plugin found for this hook spec
                # or named plugin if provided
                outpath = hook_caller(
                    _plugin=plugin_name, path=full_path, data=data, meta=meta
                )
                written.append(outpath)
            for fname in os.listdir(tmp):
                shutil.move(os.path.join(tmp, fname), path)
    except Exception as exc:
        if not already_existed:
            shutil.rmtree(path, ignore_errors=True)
        raise exc
    return written

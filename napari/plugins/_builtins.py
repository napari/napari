"""
Internal napari hook implementations to be registered by the plugin manager
"""
import os
from typing import Any, List, Optional, Union

import numpy as np
from pluggy import HookimplMarker

from ..types import (
    FullLayerData,
    ReaderFunction,
    WriterFunction,
    image_reader_to_layerdata_reader,
)
from ..utils.io import imsave, magic_imread, write_csv

napari_hook_implementation = HookimplMarker("napari")


@napari_hook_implementation(trylast=True)
def napari_get_reader(path: Union[str, List[str]]) -> ReaderFunction:
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
    return image_reader_to_layerdata_reader(magic_imread)


@napari_hook_implementation(trylast=True)
def napari_write_image(path: str, data: Any, meta: dict) -> bool:
    """Our internal fallback image writer at the end of the plugin chain.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If meta['rgb'] is `True` then the
        data should be interpreted as RGB or RGBA. If meta['is_pyramid'] is
        True, then the data should be interpreted as an image pyramid.
    meta : dict
        Image metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """
    imsave(path, data)
    return True


@napari_hook_implementation(trylast=True)
def napari_write_points(path: str, data: Any, meta: dict) -> bool:
    """Our internal fallback points writer at the end of the plugin chain.

    Append `.csv` extension to the filename if it is not already there.

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
    bool : Return True if data is successfully written.
    """
    ext = os.path.splitext(path)[1]
    if ext == '':
        path = path + '.csv'
    elif ext != '.csv':
        raise ValueError('If an extension is provided then it must be `.csv`')

    # construct table from data
    column_names = ['axis-' + str(n) for n in range(data.shape[1])]
    if bool(meta['properties']):
        column_names += meta['properties'].keys()
        prop_table = np.concatenate(
            [np.expand_dims(p, axis=1) for p in meta['properties'].values()],
            axis=1,
        )
        table = np.concatenate([data, prop_table], axis=1)
    else:
        table = data

    # add index of each point
    column_names = ['index'] + column_names
    indices = np.expand_dims(list(range(data.shape[0])), axis=1)
    table = np.concatenate([indices, table], axis=1)

    # write table to csv file
    write_csv(path, table, column_names)
    return True


@napari_hook_implementation(trylast=True)
def napari_get_writer(
    path: str, layer_types: List[str]
) -> Optional[WriterFunction]:
    """Our internal fallback file writer at the end of the writer plugin chain.

    This will create a new folder from the path and call `napari_write_<layer>`
    for each layer using the `layer.name` variable to modify the path such that
    the layers are written to unique files in the folder.

    Parameters
    ----------
    path : str
        path to file/directory

    Returns
    -------
    callable
        function that accepts the path and a list of layer_data (where
        layer_data is (data, meta, layer_type)) and writes each layer.
    """
    if os.path.exists(path):
        # If something exists at the current path return None
        return None
    else:
        return write_layer_data_with_plugins


def write_layer_data_with_plugins(
    path: str, layer_data: List[FullLayerData]
) -> bool:
    """Write layer data out into a folder one layer at a time.

    Call `napari_write_<layer>` for each layer using the `layer.name` variable
    to modify the path such that the layers are written to unique files in the
    folder.

    Parameters
    ----------
    path : str
        path to file/directory
    layer_data : list of napari.types.LayerData
        List of layer_data, where layer_data is (data, meta, layer_type).

    Returns
    -------
    bool
        Return True if data is successfully written.
    """
    from . import plugin_manager as napari_plugin_manager

    # Try and make directory based on current path
    os.makedirs(path)

    # Loop through data for each layer
    for layer_data_tuple in layer_data:
        data, meta, layer_type = layer_data_tuple
        # Get hook specification according to layer type
        hook_specification = getattr(
            napari_plugin_manager.hook, f'napari_write_{layer_type}'
        )
        # Create full path using name of layer
        full_path = os.path.join(path, meta['name'])

        # Write out data using first plugin found for this hook spec
        hook_specification(path=full_path, data=data, meta=meta)

    return True

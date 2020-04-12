"""
Internal napari hook implementations to be registered by the plugin manager
"""
import os
from typing import List, Union

from pluggy import HookimplMarker

from ..types import (
    ReaderFunction,
    image_reader_to_layerdata_reader,
    WriterFunction,
)
from ..utils.io import magic_imread
from .io import write_layer_data_with_plugins


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
def napari_get_writer(path: str, layer_types: List[str]) -> WriterFunction:
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
    os.mkdirs(path)
    return write_layer_data_with_plugins

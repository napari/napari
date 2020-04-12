"""
Internal napari hook implementations to be registered by the plugin manager
"""
# import os
from typing import List, Union, Any

from pluggy import HookimplMarker

from ..types import (
    ReaderFunction,
    image_reader_to_layerdata_reader,
    #    WriterFunction,
    #    LayerData,
)
from ..utils.io import magic_imread, imsave

# from . import plugin_manager as napari_plugin_manager


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
    """
    imsave(path, data)
    return True


# @napari_hook_implementation(trylast=True)
# def napari_get_writer(path: str, layer_types: List[str]) -> WriterFunction:
#     """Our internal fallback file writer at the end of the writer plugin chain.
#
#     This will create a new folder from the path and call `napari_write_<layer>`
#     for each layer using the `layer.name` variable to modify the path such that
#     the layers are written to unique files in the folder.
#
#     Parameters
#     ----------
#     path : str
#         path to file/directory
#
#     Returns
#     -------
#     callable
#         function that accepts the path and a list of layer_data (where
#         layer_data is (data, meta, layer_type)) and writes each layer.
#     """
#     os.mkdirs(path)
#     return write_layer_data_with_plugins
#
#
# def write_layer_data_with_plugins(path: str, layer_data: List[LayerData]):
#     """Write layer data out into a folder one layer at a time.
#
#     Call `napari_write_<layer>` for each layer using the `layer.name` variable
#     to modify the path such that the layers are written to unique files in the
#     folder.
#
#     Parameters
#     ----------
#     path : str
#         path to file/directory
#     layer_data : list of napari.types.LayerData
#         List of layer_data, where layer_data is (data, meta, layer_type).
#
#     Returns
#     -------
#     bool
#         Return True if data is successfully written.
#     """
#     # Loop through data for each layer
#     for ld in layer_data:
#         # Get hook specification according to layer type
#         hook_specification = getattr(
#             napari_plugin_manager.hook, 'napari_write_' + ld[2]
#         )
#         # Create full path using name of layer
#         full_path = os.path.join(path, ld[0]['name'])
#
#         # Write out data using first plugin found for this hook spec
#         hook_specification(path=full_path, data=ld[0], meta=ld[1])
#
#     return True

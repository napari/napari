"""
Internal napari hook implementations to be registered by the plugin manager
"""
from typing import List, Union

from pluggy import HookimplMarker

from ..types import ReaderFunction, image_reader_to_layerdata_reader
from ..utils.io import magic_imread

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

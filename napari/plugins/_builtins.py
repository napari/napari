"""
Internal napari hook implementations to be registered by the plugin manager
"""
from pluggy import HookimplMarker
from typing import Union, List
from ..types import ReaderFunction, LayerData
from ..utils import io

napari_hook_implementation = HookimplMarker("napari")


def _interal_reader_plugin(path: Union[str, List[str]]) -> List[LayerData]:
    """Pass ``path`` to our magic_imread function and return as LayerData."""
    return [(io.magic_imread(path, stack=not isinstance(path, str)),)]


@napari_hook_implementation(trylast=True)
def napari_get_reader(path: str) -> ReaderFunction:
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
    return _interal_reader_plugin

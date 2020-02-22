"""
Internal napari hook implementations to be registered by the plugin manager
"""
from pluggy import HookimplMarker

napari_hook_implementation = HookimplMarker("napari")


@napari_hook_implementation(trylast=True)
def napari_get_reader(path: str):
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
    return lambda path: [(None, {'path': path})]

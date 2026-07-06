"""napari.components provides the public-facing models for widgets
and other utilities that the user will be able to programmatically interact
with.

Classes
-------
Dims
    Current indices along each data dimension, together with which dimensions
    are being displayed, projected, sliced...
LayerList
    List of layers currently present in the viewer.
ViewerModel
    Data viewer displaying the currently rendered scene and
    layer-related controls.
"""

# Note that importing _viewer_key_bindings is needed as the Viewer gets
# decorated with keybindings during that process
from napari.components._viewer_key_bindings import ViewerModel
from napari.components.camera import Camera
from napari.components.dims import Dims
from napari.components.layerlist import LayerList

__all__ = ['Camera', 'Dims', 'LayerList', 'ViewerModel']

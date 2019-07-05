"""The elements module provides the public-facing GUI widgets, windows,
and other utilities that the user will be able to programmatically interact
with.

Classes
-------
Dims
    Current indices along each data dimension, together with which dimensions
    are being displayed, projected, sliced...
LayersList
    List of layers currently present in the viewer.
ViewerModel
    Data viewer displaying the currently rendered scene and
    layer-related controls.
"""
from .viewer_model import ViewerModel
from .layers_list import LayersList
from .dims import Dims

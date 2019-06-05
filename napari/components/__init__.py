"""The elements module provides the public-facing GUI widgets, windows,
and other utilities that the user will be able to programmatically interact
with.

Classes
-------
Window
    Window containing file menus, toolbars, and viewers.
ViewerModel
    Data viewer displaying the currently rendered scene and
    layer-related controls.
"""
from ._window import Window, QtApplication
from ._viewer import ViewerModel
from ._layers_list import LayersList
from ._dims import Dims

from napari.layers.shapes import _shapes_key_bindings
from napari.layers.shapes.shapes import Shapes

# Note that importing _shapes_key_bindings is needed as the Shapes layer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
del _shapes_key_bindings


__all__ = ['Shapes']

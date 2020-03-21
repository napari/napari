from .shapes import Shapes
from . import _shapes_key_bindings

# Note that importing _viewer_key_bindings is needed as the Viewer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and is deleted below
del _shapes_key_bindings

from . import _image_key_bindings
from .image import Image

# Note that importing _image_key_bindings is needed as the Image layer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
del _image_key_bindings

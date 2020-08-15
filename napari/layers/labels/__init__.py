from . import _labels_key_bindings
from .labels import Labels

# Note that importing _labels_key_bindings is needed as the Labels layer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
del _labels_key_bindings

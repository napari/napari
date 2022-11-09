import warnings

from napari.settings import *
from napari.utils.translations import trans

warnings.warn(
    trans._(
        "'napari.utils.settings' has moved to 'napari.settings' in 0.4.11. This will raise an ImportError in a future version",
        deferred=True,
    ),
    FutureWarning,
    stacklevel=2,
)

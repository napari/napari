import warnings

from napari.settings import *  # noqa: F403

warnings.warn(
    "'napari.utils.settings' has moved to 'napari.settings' in 0.4.11. This will raise an ImportError in a future version",
    FutureWarning,
    stacklevel=2,
)

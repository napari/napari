import warnings

from napari.settings import *  # noqa: F403


warnings.warn(
    trans._(
        "'napari.utils.settings' has moved to 'napari.settings' in 0.4.11. This will raise an ImportError in a future version",
        deferred=True,
    ),
    FutureWarning,
    stacklevel=2,
)

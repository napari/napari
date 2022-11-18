import warnings

from napari.utils.translations import trans

warnings.warn(
    trans._(
        'progress has moved from qt since 0.4.11. Use `from napari.utils import progress` instead',
        deferred=True,
    ),
    category=FutureWarning,
    stacklevel=3,
)

from napari.utils import progrange, progress  # noqa

__all__ = ('progress', 'progrange')

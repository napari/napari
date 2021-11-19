import warnings

from ..utils.translations import trans

warnings.warn(
    trans._(
        'progress has moved from qt. Use `from napari.utils import progress` instead',
        deferred=True,
    ),
    category=FutureWarning,
    stacklevel=3,
)

from ..utils import progrange, progress  # noqa

__all__ = ('progress', 'progrange')

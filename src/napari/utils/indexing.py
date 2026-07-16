import warnings

from napari.utils._indexing import index_in_slice

__all__ = ['index_in_slice']

warnings.warn(
    'napari.utils.indexing is deprecated since 0.4.19 and will be removed in 0.5.0.',
    FutureWarning,
    stacklevel=2,
)

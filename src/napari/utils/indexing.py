from napari.utils._indexing import index_in_slice
from napari.utils.migrations import deprecation_warning

__all__ = ['index_in_slice']

deprecation_warning(
    name='The `napari.utils.indexing` module',
    since='0.4.19',
    window='2027-Q1',
    details=(
        'Its only public member, `index_in_slice`, now lives in the private '
        '`napari.utils._indexing` module.'
    ),
)

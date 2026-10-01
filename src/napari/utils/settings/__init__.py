from napari.settings import *  # noqa: F403
from napari.utils.migrations import deprecation_warning

deprecation_warning(
    name='The `napari.utils.settings` module',
    replacement='`napari.settings`',
    since='0.4.11',
    window='2027-Q1',
    details='Once removed, importing it will raise ImportError.',
)

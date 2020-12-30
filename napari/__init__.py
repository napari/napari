try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"


from .viewer import Viewer  # isort:skip

# this unused import is here to fix a very strange bug.
# there is some mysterious magical goodness in scipy stats that needs
# to be imported early.
# see: https://github.com/napari/napari/issues/925
# see: https://github.com/napari/napari/issues/1347
from scipy import stats  # noqa: F401

from ._event_loop import event_loop
from .plugins.io import save_layers


def __getattr__(name):
    if name == 'gui_qt':
        from warnings import warn

        warn(
            "napari.gui_qt is deprecated. Use napari.event_loop instead",
            DeprecationWarning,
        )
        return event_loop
    raise AttributeError(f"module {__name__} has no attribute {name}")


# register napari object types with magicgui if it is installed
from .utils import _magicgui, sys_info
from .view_layers import (  # type: ignore
    view_image,
    view_labels,
    view_path,
    view_points,
    view_shapes,
    view_surface,
    view_tracks,
    view_vectors,
)

_magicgui.register_types_with_magicgui()
del _magicgui


del stats

__all__ = [
    'Viewer',
    'save_layers',
    'sys_info',
    'view_image',
    'view_labels',
    'view_path',
    'view_points',
    'view_shapes',
    'view_surface',
    'view_tracks',
    'view_vectors',
    'event_loop',
]

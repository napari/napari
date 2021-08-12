import os
import sys

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

# Allows us to use pydata/sparse arrays as layer data
os.environ.setdefault('SPARSE_AUTO_DENSIFY', '1')
del os

# Add everything that needs to be accessible from the napari namespace here.
_proto_all_ = [
    '__version__',
    'components',
    'experimental',
    'layers',
    'qt',
    'types',
    'viewer',
    'utils',
]

_submod_attrs = {
    '_event_loop': ['gui_qt', 'run'],
    'plugins.io': ['save_layers'],
    'utils': ['sys_info'],
    'utils.notifications': ['notification_manager'],
    'view_layers': [
        'view_image',
        'view_labels',
        'view_path',
        'view_points',
        'view_shapes',
        'view_surface',
        'view_tracks',
        'view_vectors',
    ],
    'viewer': ['Viewer'],
}

# All imports in __init__ are hidden inside of `__getattr__` to prevent
# importing the full chain of packages required when calling `import napari`.
#
# This has the biggest implications for running `napari` on the command line
# (or running `python -m napari`) since `napari.__init__` gets imported
# on the way to `napari.__main__`. Importing everything here has the
# potential to take a second or more, so we definitely don't want to import it
# just to access the CLI (which may not actually need any of the imports)

from ._lazy import install_lazy

__getattr__, __dir__, __all__ = install_lazy(
    __name__, _proto_all_, _submod_attrs
)
del install_lazy


class AttributeProtector:
    def __init__(self, module):
        self.__dict__['module'] = module

    def __getattr__(self, attr):
        return getattr(self.module, attr)

    def __setattr__(self, attr, val):
        if attr == 'current_viewer':
            raise TypeError(f'{attr!r} can not be set')


sys.modules[__name__] = AttributeProtector(sys.modules[__name__])

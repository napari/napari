import os

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"

# Allows us to use pydata/sparse arrays as layer data
os.environ.setdefault('SPARSE_AUTO_DENSIFY', '1')

# Add everything that needs to be accessible from the napari namespace here.
__all__ = [
    '__version__',
    'components',
    'experimental',
    'gui_qt',
    'layers',
    'notification_manager',
    'qt',
    'run',
    'save_layers',
    'sys_info',
    'utils',
    'view_image',
    'view_labels',
    'view_path',
    'view_points',
    'view_shapes',
    'view_surface',
    'view_tracks',
    'view_vectors',
    'Viewer',
]


# All imports in __init__ are hidden inside of `__getattr__` to prevent
# importing the full chain of packages required when calling `import napari`.
#
# This has the biggest implications for running `napari` on the command line
# (or running `python -m napari`) since `napari.__init__` gets imported
# on the way to `napari.__main__`. Importing everything here has the
# potential to take a second or more, so we definitely don't want to import it
# just to access the CLI (which may not actually need any of the imports)

from ._lazy import install_lazy

__getattr__, __dir__, _ = install_lazy(__name__, __all__)
del install_lazy

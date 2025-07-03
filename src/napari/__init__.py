import os

from lazy_loader import attach as _attach

from napari._check_numpy_version import limit_numpy1x_threads_on_macos_arm

try:
    from napari._version import version as __version__
except ImportError:
    __version__ = 'not-installed'

# Allows us to use pydata/sparse arrays as layer data
os.environ.setdefault('SPARSE_AUTO_DENSIFY', '1')
limit_numpy1x_threads_on_macos_arm()


def _check_installation_path():  # pragma: no cover
    """Check for installation path conflicts.

    Check if napari is present in site-packages. If napari is installed in editable mode,
    notify the user of a the conflict that napari is also in site-packages.
    """
    import sys
    from pathlib import Path

    if 'pytest' in sys.modules:
        # pytest is running, skip the check
        return

    napari_installation_path = Path(__file__).absolute().parent.parent
    if napari_installation_path.name == 'site-packages':
        # napari is installed in non-editable mode
        return

    import numpy as np

    # Use numpy location to determine a site-packages path
    site_packages_path = Path(np.__file__).absolute().parent.parent
    if site_packages_path.name != 'site-packages':
        # numpy is not installed in site-packages
        return

    napari_site_packages_path = site_packages_path / 'napari'
    napari_builtins_package_path = site_packages_path / 'napari_builtins'

    path_text = ''
    if napari_site_packages_path.exists():
        path_text += (
            f'Path to a napari directory: {napari_site_packages_path}.\n'
        )
    if napari_builtins_package_path.exists():
        path_text += f'Path to a napari_builtins directory: {napari_builtins_package_path}.\n'

    if path_text:
        text = (
            'Mix of local and non local installation detected.\n'
            'Napari is installed in editable mode but also found napari '
            'directory in site-packages.\n'
            f'{path_text}'
            'Mix of local and non local installation is leading '
            'to hard to understand errors. '
            'See https://napari.org/stable/troubleshooting.html#mixed-napari-installations for more details.'
        )
        raise RuntimeError(text)


_check_installation_path()

del limit_numpy1x_threads_on_macos_arm
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
    '_event_loop': ['run'],
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
        'imshow',
    ],
    'viewer': ['Viewer', 'current_viewer'],
}

# All imports in __init__ are hidden inside of `__getattr__` to prevent
# importing the full chain of packages required when calling `import napari`.
#
# This has the biggest implications for running `napari` on the command line
# (or running `python -m napari`) since `napari.__init__` gets imported
# on the way to `napari.__main__`. Importing everything here has the
# potential to take a second or more, so we definitely don't want to import it
# just to access the CLI (which may not actually need any of the imports)


__getattr__, __dir__, __all__ = _attach(
    __name__, submodules=_proto_all_, submod_attrs=_submod_attrs
)
del _attach

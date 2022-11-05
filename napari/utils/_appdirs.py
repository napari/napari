import os
import sys
from functools import partial
from typing import Callable, Optional

import appdirs

from napari._version import __version_tuple__

_appname = 'napari'
_appauthor = False

version_string = '.'.join(str(x) for x in __version_tuple__[:3])


# all of these also take an optional "version" argument ... but if we want
# to be able to update napari while using data (e.g. plugins, settings) from
# an earlier version, we should leave off the version.

user_data_dir: Callable[[], str] = partial(
    appdirs.user_data_dir, _appname, _appauthor
)
user_config_dir: Callable[[], str] = partial(
    appdirs.user_config_dir, _appname, _appauthor, version_string
)
user_cache_dir: Callable[[], str] = partial(
    appdirs.user_cache_dir, _appname, _appauthor, version_string
)
user_state_dir: Callable[[], str] = partial(
    appdirs.user_state_dir, _appname, _appauthor
)
user_log_dir: Callable[[], str] = partial(
    appdirs.user_log_dir, _appname, _appauthor
)


def user_plugin_dir() -> str:
    """Prefix directory for external pip install.

    Suitable for use as argument with `pip install --prefix`.
    On mac and windows, we can install directly into the bundle.  This may be
    used on Linux to pip install packages outside of the bundle with:
    ``pip install --prefix user_plugin_dir()``
    """
    return os.path.join(user_data_dir(), 'plugins')


def user_site_packages() -> str:
    """Platform-specific location of site-packages folder in user library"""
    if os.name == 'nt':
        return os.path.join(user_plugin_dir(), 'Lib', 'site-packages')

    python_dir = f'python{sys.version_info.major}.{sys.version_info.minor}'
    return os.path.join(user_plugin_dir(), 'lib', python_dir, 'site-packages')


def bundled_site_packages() -> Optional[str]:
    """Platform-specific location of site-packages folder in bundles."""
    exe_dir = os.path.dirname(sys.executable)
    if os.name == 'nt':
        return os.path.join(exe_dir, "Lib", "site-packages")

    if sys.platform.startswith('darwin'):
        python_dir = f'python{sys.version_info.major}.{sys.version_info.minor}'
        return os.path.join(
            os.path.dirname(exe_dir), "lib", python_dir, "site-packages"
        )

    # briefcase linux bundles cannot install into the AppImage
    return None

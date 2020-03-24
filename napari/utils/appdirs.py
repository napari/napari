r"""Platform specific directories for storing napari data.

Typical user_data_dir:
    Mac OS X:   ~/Library/Application Support/napari
    Unix:       ~/.local/share/napari    # or in $XDG_DATA_HOME, if defined
    Win:        C:\Users\<username>\AppData\Local\napari

Typical user_config_dir:
    Mac OS X:   ~/Library/Preferences/napari
    Unix:       ~/.config/napari     # or in $XDG_CONFIG_HOME, if defined
    Win:        same as user_data_dir

Typical user_cache_dir:
    Mac OS X:   ~/Library/Caches/napari
    Unix:       ~/.cache/napari (XDG default)
    Win:        C:\Users\<username>\AppData\Local\napari\Cache

Typical user_state_dir:
    Mac OS X:   same as user_data_dir
    Unix:       ~/.local/state/napari   # or in $XDG_STATE_HOME, if defined
    Win:        same as user_data_dir

Typical user_log_dir:
    Mac OS X:   ~/Library/Logs/napari
    Unix:       ~/.cache/napari/log  # or under $XDG_CACHE_HOME if defined
    Win:        C:\Users\<username>\AppData\Local\napari\Logs
"""
from functools import partial
from os.path import join
from sys import version_info

from . import _appdirs

_appname = 'napari'
_appauthor = False

# all of these also take an optional "version" argument ... but if we want
# to be able to update napari while using data (e.g. plugins, settings) from
# an earlier version, we should leave off the version.

user_data_dir = partial(_appdirs.user_data_dir, _appname, _appauthor)
user_config_dir = partial(_appdirs.user_config_dir, _appname, _appauthor)
user_cache_dir = partial(_appdirs.user_cache_dir, _appname, _appauthor)
user_state_dir = partial(_appdirs.user_state_dir, _appname, _appauthor)
user_log_dir = partial(_appdirs.user_log_dir, _appname, _appauthor)


def user_plugin_dir() -> str:
    return join(user_data_dir(), 'plugins')


def user_site_packages() -> str:
    # TODO: does this work cross platform?
    python_dir = f'python{version_info.major}.{version_info.minor}'
    return join(user_plugin_dir(), 'lib', python_dir, 'site-packages')

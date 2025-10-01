import hashlib
import os
import sys
from collections.abc import Callable
from functools import partial

import appdirs

__all__ = (
    'user_cache_dir',
    'user_config_dir',
    'user_data_dir',
    'user_log_dir',
    'user_state_dir',
)

PREFIX_PATH = os.path.realpath(sys.prefix)

UV_POSSIBLE_PATH = appdirs.user_cache_dir('uv')

if PREFIX_PATH.startswith(UV_POSSIBLE_PATH):
    environment_marker = 'uvx'
else:
    environment_marker = f'{os.path.basename(PREFIX_PATH)}_{hashlib.sha1(PREFIX_PATH.encode()).hexdigest()}'

_appname = 'napari'
_appauthor = False


# all of these also take an optional "version" argument ... but if we want
# to be able to update napari while using data (e.g. plugins, settings) from
# an earlier version, we should leave off the version.

user_data_dir: Callable[[], str] = partial(
    appdirs.user_data_dir, _appname, _appauthor
)
user_config_dir: Callable[[], str] = partial(
    appdirs.user_config_dir, _appname, _appauthor, environment_marker
)
user_cache_dir: Callable[[], str] = partial(
    appdirs.user_cache_dir, _appname, _appauthor, environment_marker
)
user_state_dir: Callable[[], str] = partial(
    appdirs.user_state_dir, _appname, _appauthor
)
user_log_dir: Callable[[], str] = partial(
    appdirs.user_log_dir, _appname, _appauthor
)

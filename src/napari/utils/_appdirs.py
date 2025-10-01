import hashlib
import os
import shutil
import sys
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path

import appdirs
from packaging.version import InvalidVersion, parse as parse_version

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
    environment_marker = os.path.join(
        'uvx', parse_version(version('napari')).base_version
    )
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


def _maybe_migrate_uvx_settings() -> bool:
    """If we are in an uv environment, and there are no settings in the
    current config dir, but there are settings in the uvx config dir,
    move them over.

    Returns
    -------
    bool
        Whether or not a migration was performed.
    """
    if not environment_marker.startswith('uvx'):
        return False  # only migrate if we are in an uvx environment

    config_path = Path(user_config_dir())
    if config_path.exists():
        return False  # nothing to do, the current config path already exists

    base_config_path = config_path.parent

    if not base_config_path.exists():
        return False  # nothing to do, the base config path doesn't exist

    napari_version = parse_version(version('napari'))
    older_versions = []

    for fname in base_config_path.iterdir():
        try:
            dir_version = parse_version(fname.name)
        except InvalidVersion:
            continue
        if dir_version < napari_version:
            older_versions.append((dir_version, fname))

    if not older_versions:
        return False  # nothing to do, there are no older versions

    # get the latest version that is older than the current version
    shutil.copytree(max(older_versions)[1], config_path)
    return True

import re
from fnmatch import fnmatch
from typing import Set, Union

from npe2 import PluginManifest

from napari.settings import get_settings

from . import _npe2


def get_preferred_reader(_path):
    """Return preferred reader for _path from settings, if one exists."""
    reader_settings = get_settings().plugins.extension2reader
    for pattern, reader in reader_settings.items():
        # TODO: we return the first one we find - more work should be done here
        # in case other patterns would match - do we return the most specific?
        if fnmatch(_path, pattern):
            return reader


def normalized_name(name: str) -> str:
    """
    Normalize a plugin name by replacing underscores and dots by dashes and
    lower casing it.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def get_filename_patterns_for_reader(plugin_name: str):
    """Return recognized filename patterns, if any, for a given plugin.

    Where a plugin provides multiple readers it will return a set of
    all recognized filename patterns.

    Parameters
    ----------
    plugin_name : str
        name of plugin to find filename patterns for

    Returns
    -------
    set
        set of filename patterns accepted by all plugin's reader contributions
    """
    all_fn_patterns: Set[str] = set()
    current_plugin: Union[PluginManifest, None] = None
    for manifest in _npe2.iter_manifests():
        if manifest.name == plugin_name:
            current_plugin = manifest
    if current_plugin:
        readers = current_plugin.contributions.readers or []
        for reader in readers:
            all_fn_patterns = all_fn_patterns.union(
                set(reader.filename_patterns)
            )
    return all_fn_patterns

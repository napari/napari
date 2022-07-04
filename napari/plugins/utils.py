import re
from fnmatch import fnmatch
from typing import Dict, Set, Tuple, Union

from npe2 import PluginManifest

from napari.settings import get_settings

from . import _npe2, plugin_manager


def get_preferred_reader(_path):
    """Return preferred reader for _path from settings, if one exists."""
    reader_settings = get_settings().plugins.extension2reader
    for pattern, reader in reader_settings.items():
        # TODO: we return the first one we find - more work should be done here
        # in case other patterns would match - do we return the most specific?
        if fnmatch(_path, pattern):
            return reader


def get_potential_readers(filename: str) -> Dict[str, str]:
    """Given filename, returns all readers that may read the file.

    Original plugin engine readers are checked based on returning
    a function from `napari_get_reader`. Npe2 readers are iterated
    based on file extension and accepting directories.

    Returns
    -------
    Dict[str, str]
        dictionary of registered name to display_name
    """
    readers = {}
    hook_caller = plugin_manager.hook.napari_get_reader
    for impl in hook_caller.get_hookimpls():
        reader = hook_caller._call_plugin(impl.plugin_name, path=filename)
        if callable(reader):
            readers[impl.plugin_name] = impl.plugin_name
    readers.update(_npe2.get_readers(filename))
    return readers


def get_all_readers() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return a dict of all npe2 readers and one of all npe1 readers

    Can be removed once npe2 shim is activated.
    """

    npe2_readers = _npe2.get_readers()

    npe1_readers = {}
    for spec, hook_caller in plugin_manager.hooks.items():
        if spec == 'napari_get_reader':
            potential_readers = hook_caller.get_hookimpls()
            for get_reader in potential_readers:
                npe1_readers[get_reader.plugin_name] = get_reader.plugin_name

    return npe2_readers, npe1_readers


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
    # npe1 plugins
    else:
        _, npe1_readers = get_all_readers()
        if plugin_name in npe1_readers:
            all_fn_patterns = {'*'}

    return all_fn_patterns

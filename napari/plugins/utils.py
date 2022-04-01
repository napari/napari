import os
import re
from typing import Dict

from napari.settings import get_settings

from . import _npe2, plugin_manager


def get_preferred_reader(_path):
    """Return preferred reader for _path from settings, if one exists."""
    _, extension = os.path.splitext(_path)
    if extension:
        reader_settings = get_settings().plugins.extension2reader
        return reader_settings.get(extension)


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
    readers = _npe2.get_readers(filename)

    npe1_readers = {}
    for spec, hook_caller in plugin_manager.hooks.items():
        if spec == 'napari_get_reader':
            potential_readers = hook_caller.get_hookimpls()
            for get_reader in potential_readers:
                reader = hook_caller._call_plugin(
                    get_reader.plugin_name, path=filename
                )
                if callable(reader):
                    npe1_readers[
                        get_reader.plugin_name
                    ] = get_reader.plugin_name
    readers.update(npe1_readers)

    # if npe2 is present, remove npe1 builtins
    if 'napari' in readers and 'builtins' in readers:
        del readers['builtins']

    return readers


def normalized_name(name: str) -> str:
    """
    Normalize a plugin name by replacing underscores and dots by dashes and
    lower casing it.
    """
    return re.sub(r"[-_.]+", "-", name).lower()

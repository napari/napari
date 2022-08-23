import os
import os.path as osp
import re
from enum import IntFlag
from fnmatch import fnmatch
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from npe2 import PluginManifest

from napari.settings import get_settings

from . import _npe2, plugin_manager


class MatchFlag(IntFlag):
    NONE = 0
    SET = 1
    ANY = 2
    STAR = 4


@lru_cache
def score_specificity(pattern: str) -> Tuple[bool, int, List[MatchFlag]]:
    """Score an fnmatch pattern, with higher specificities having lower scores.

    Absolute paths have highest specificity,
    followed by paths with the most nesting,
    then by path segments with the least ambiguity.

    Parameters
    ----------
    pattern : str
        Pattern to score.

    Returns
    -------
    relpath : boolean
        Whether the path is relative or absolute.
    nestedness : negative int
        Level of nestedness of the path, lower is deeper.
    score : List[MatchFlag]
        Path segments scored by ambiguity, higher score is higher ambiguity.
    """
    pattern = osp.normpath(pattern)

    segments = pattern.split(osp.sep)
    score: List[MatchFlag] = []
    ends_with_star = False

    def add(match_flag):
        score[-1] |= match_flag

    # built-in fnmatch does not allow you to escape meta-characters
    # so we don't need to handle them :)
    for segment in segments:
        # collapse foo/*/*/*.bar or foo*/*.bar but not foo*bar/*.baz
        if segment and not (ends_with_star and segment.startswith('*')):
            score.append(MatchFlag.NONE)

        if '*' in segment:
            add(MatchFlag.STAR)
        if '?' in segment:
            add(MatchFlag.ANY)
        if '[' in segment and ']' in segment[segment.index('[') :]:
            add(MatchFlag.SET)

        ends_with_star = segment.endswith('*')

    return not osp.isabs(pattern), 1 - len(score), score


def _get_preferred_readers(path: str) -> Iterable[Tuple[str, str]]:
    """Given filepath, find matching readers from preferences.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    filtered_preferences : Iterable[Tuple[str, str]]
        Filtered patterns and their corresponding readers.
    """

    if osp.isdir(path):
        if not path.endswith(os.sep):
            path = path + os.sep

    reader_settings = get_settings().plugins.extension2reader
    print('reader settings', reader_settings)
    print(
        'result',
        str(filter(lambda kv: fnmatch(path, kv[0]), reader_settings.items())),
    )
    for kv in reader_settings.items():
        print('matching', path, kv, fnmatch(path, kv[0]))
    return filter(lambda kv: fnmatch(path, kv[0]), reader_settings.items())


def get_preferred_reader(path: str) -> Optional[str]:
    """Given filepath, find the best matching reader from the preferences.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    reader : str or None
        Best matching reader, if found.
    """
    readers = sorted(
        _get_preferred_readers(path), key=lambda kv: score_specificity(kv[0])
    )
    print('readers', readers)
    if readers:
        preferred = readers[0]
        _, reader = preferred
        print(reader)
        return reader

    return None


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

"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Iterator, List, Tuple
from urllib import request

from npe2 import PackageMetadata

from .utils import normalized_name


@lru_cache
def pypi_plugin_summaries() -> List[Dict]:
    """Return PackageMetadata object for all known napari plugins."""
    with request.urlopen("https://npe2api.vercel.app/api/summary") as response:
        return json.load(response)


@lru_cache
def conda_map() -> List[Dict]:
    """Return PackageMetadata object for all known napari plugins."""
    with request.urlopen("https://npe2api.vercel.app/api/conda") as response:
        return json.load(response)


def iter_napari_plugin_info() -> Iterator[Tuple[PackageMetadata, bool]]:
    """Return a generator that yields ProjectInfo of available napari plugins."""
    with ThreadPoolExecutor() as executor:
        data = executor.submit(pypi_plugin_summaries)
        at_conda = executor.submit(conda_map)

    at_conda = at_conda.result()
    for info in data.result():
        # TODO: use this better.
        # this would require changing the api that qt_plugin_dialog expects to receive
        # (and it doesn't currently receive this from the hub API)
        info.pop("display_name", None)  # TODO, use this
        name = info.pop("name")
        meta = PackageMetadata(name=normalized_name(name), **info)
        yield meta, name in at_conda

"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple, TypedDict, cast
from urllib import request

from npe2 import PackageMetadata

from .utils import normalized_name

PyPIname = str


class SummaryDict(TypedDict):
    """Objects returned at https://npe2api.vercel.app/api/summary ."""

    name: PyPIname
    version: str
    display_name: str
    summary: str
    author: str
    license: str
    home_page: str


@lru_cache
def pypi_plugin_summaries() -> List[SummaryDict]:
    """Return PackageMetadata object for all known napari plugins."""
    with request.urlopen("https://npe2api.vercel.app/api/summary") as response:
        return json.load(response)


@lru_cache
def conda_map() -> Dict[PyPIname, Optional[str]]:
    """Return map of PyPI package name to conda_channel/package_name ()."""
    with request.urlopen("https://npe2api.vercel.app/api/conda") as response:
        return json.load(response)


def iter_napari_plugin_info() -> Iterator[Tuple[PackageMetadata, bool]]:
    """Iterator of tuples of ProjectInfo, Conda availability for all napari plugins."""
    with ThreadPoolExecutor() as executor:
        data = executor.submit(pypi_plugin_summaries)
        _conda = executor.submit(conda_map)

    conda = _conda.result()
    for info in data.result():
        _info = cast(Dict[str, str], dict(info))
        # TODO: use this better.
        # this would require changing the api that qt_plugin_dialog expects to
        # receive (and it doesn't currently receive this from the hub API)
        _info.pop("display_name", None)

        # TODO: I'd prefer we didn't normalize the name here, but it's needed for
        # parity with the hub api.  change this later.
        name = _info.pop("name")
        meta = PackageMetadata(name=normalized_name(name), **_info)
        yield meta, (name in conda)

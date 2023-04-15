"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypedDict,
    cast,
)
from urllib.request import Request, urlopen

from npe2 import PackageMetadata
from typing_extensions import NotRequired

from napari.plugins.utils import normalized_name

PyPIname = str


@lru_cache
def _user_agent() -> str:
    """Return a user agent string for use in http requests."""
    import platform

    from napari import __version__
    from napari.utils import misc

    if misc.running_as_bundled_app():
        env = 'briefcase'
    elif misc.running_as_constructor_app():
        env = 'constructor'
    elif misc.in_jupyter():
        env = 'jupyter'
    elif misc.in_ipython():
        env = 'ipython'
    else:
        env = 'python'

    parts = [
        ('napari', __version__),
        ('runtime', env),
        (platform.python_implementation(), platform.python_version()),
        (platform.system(), platform.release()),
    ]
    return ' '.join(f'{k}/{v}' for k, v in parts)


class SummaryDict(TypedDict):
    """Objects returned at https://npe2api.vercel.app/api/extended_summary ."""

    name: NotRequired[PyPIname]
    version: str
    display_name: NotRequired[str]
    summary: str
    author: str
    license: str
    home_page: str
    pypi_versions: NotRequired[List[str]]
    conda_versions: NotRequired[List[str]]


@lru_cache
def plugin_summaries() -> List[SummaryDict]:
    """Return PackageMetadata object for all known napari plugins."""
    url = "https://npe2api.vercel.app/api/extended_summary"
    with urlopen(Request(url, headers={'User-Agent': _user_agent()})) as resp:
        return json.load(resp)


@lru_cache
def conda_map() -> Dict[PyPIname, Optional[str]]:
    """Return map of PyPI package name to conda_channel/package_name ()."""
    url = "https://npe2api.vercel.app/api/conda"
    with urlopen(Request(url, headers={'User-Agent': _user_agent()})) as resp:
        return json.load(resp)


class _ExtraDict(TypedDict):
    home_page: str
    pypi_versions: List[str]
    conda_versions: List[str]


def iter_napari_plugin_info() -> (
    Iterator[Tuple[PackageMetadata, bool, _ExtraDict]]
):
    """Iterator of tuples of ProjectInfo, Conda availability for all napari plugins."""
    with ThreadPoolExecutor() as executor:
        data = executor.submit(plugin_summaries)
        _conda = executor.submit(conda_map)

    conda = _conda.result()
    for info in data.result():
        info_ = cast(SummaryDict, dict(info))

        # TODO: use this better.
        # this would require changing the api that qt_plugin_dialog expects to
        # receive
        info_.pop("display_name", None)

        # TODO: once the new version of npe2 is out, this can be refactored
        # to all the metadata includes the conda and pypi versions.
        extra_info = _ExtraDict(
            home_page=info_.get("home_page", ""),
            pypi_versions=info_.pop("pypi_versions"),
            conda_versions=info_.pop("conda_versions"),
        )
        info_["name"] = normalized_name(info_["name"])
        meta = PackageMetadata(**info_)

        yield meta, (info_["name"] in conda), extra_info

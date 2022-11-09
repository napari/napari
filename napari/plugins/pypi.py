"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple, TypedDict, cast
from urllib.request import Request, urlopen

from npe2 import PackageMetadata

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
    url = "https://npe2api.vercel.app/api/summary"
    with urlopen(Request(url, headers={'User-Agent': _user_agent()})) as resp:
        return json.load(resp)


@lru_cache
def conda_map() -> Dict[PyPIname, Optional[str]]:
    """Return map of PyPI package name to conda_channel/package_name ()."""
    url = "https://npe2api.vercel.app/api/conda"
    with urlopen(Request(url, headers={'User-Agent': _user_agent()})) as resp:
        return json.load(resp)


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

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

DEFAULT_CHANNEL = "conda-forge"


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
    """Objects returned at https://npe2api-5lai6hg8a-napari.vercel.app/api/extended_summary ."""

    name: PyPIname
    version: str
    display_name: str
    summary: str
    author: str
    license: str
    home_page: str


@lru_cache
def plugin_summaries() -> List[SummaryDict]:
    """Return PackageMetadata object for all known napari plugins."""
    url = "https://npe2api-5lai6hg8a-napari.vercel.app/api/extended_summary"
    with urlopen(Request(url, headers={'User-Agent': _user_agent()})) as resp:
        return json.load(resp)


@lru_cache
def conda_map() -> Dict[PyPIname, Optional[str]]:
    """Return map of PyPI package name to conda_channel/package_name ()."""
    url = "https://npe2api.vercel.app/api/conda"
    with urlopen(Request(url, headers={'User-Agent': _user_agent()})) as resp:
        return json.load(resp)


# @lru_cache
# def pypi_package_versions(package_name: str) -> List[str]:
#     """Get available versions of a package on pypi.

#     Parameters
#     ----------
#     package_name : str
#         Name of the package.

#     Returns
#     -------
#     list
#         Versions available on pypi.
#     """
#     url = f"https://pypi.org/simple/{package_name}"
#     html = requests.get(url).text
#     package_name_ = package_name.replace('-', '_')
#     ret1 = re.findall(f">{package_name}-(.+).tar", html)
#     ret2 = re.findall(f">{package_name_}-(.+).tar", html)
#     return ret1 + ret2


# @lru_cache
# def conda_package_data(
#     package_name: str, channel: str = DEFAULT_CHANNEL
# ) -> dict:
#     """Return information on package from given channel.

#     Parameters
#     ----------
#     package_name : str
#         Name of package.
#     channel : str, optional
#         Channel to search, by default `DEFAULT_CHANNEL`.

#     Returns
#     -------
#     dict
#         Package information.
#     """
#     url = f"https://api.anaconda.org/package/{channel}/{package_name}"
#     response = requests.get(url)
#     return response.json()


# @lru_cache
# def conda_package_versions(
#     package_name: str, channel: str = DEFAULT_CHANNEL
# ) -> List[str]:
#     """Return information on package from given channel.

#     Parameters
#     ----------
#     package_name : str
#         Name of package.
#     channel : str, optional
#         Channel to search, by default `DEFAULT_CHANNEL`.

#     Returns
#     -------
#     list of str
#         Package versions.
#     """
#     return conda_package_data(package_name, channel=channel).get(
#         "versions", []
#     )


def iter_napari_plugin_info() -> Iterator[
    Tuple[PackageMetadata, bool, List[str], List[str]]
]:
    """Iterator of tuples of ProjectInfo, Conda availability for all napari plugins."""
    with ThreadPoolExecutor() as executor:
        data = executor.submit(plugin_summaries)
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
        extra_info = {
            "home_page": _info.get("home_page", ""),
            "pypi_versions": _info.pop("pypi_versions"),
            "conda_versions": _info.pop("conda_versions"),
        }
        name = _info.pop("name")
        meta = PackageMetadata(name=normalized_name(name), **_info)

        yield meta, (name in conda), extra_info

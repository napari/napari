"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Generator, List, NamedTuple, Optional
from urllib import error, parse, request

PYPI_SIMPLE_API_URL = 'https://pypi.org/simple/'

setup_py_entrypoint = re.compile(
    r"entry_points\s?=\s?([^}]*napari.plugin[^}]*)}"
)
setup_py_pypi_name = re.compile(
    r"setup\s?\(.*name\s?=\s?['\"]([^'\"]+)['\"]", re.DOTALL
)


class ProjectInfo(NamedTuple):
    """Info associated with a PyPI Project."""

    name: str
    version: str
    url: str
    summary: str
    author: str
    license: str


@lru_cache(maxsize=128)
def get_packages_by_prefix(prefix: str) -> Dict[str, str]:
    """Search for packages starting with ``prefix`` on pypi.

    Packages using naming convention: http://bit.ly/pynaming-convention
    can be autodiscovered on pypi using the SIMPLE API:
    https://www.python.org/dev/peps/pep-0503/

    Returns
    -------
    dict
        {name: url} for all packages at pypi that start with ``prefix``
    """

    with request.urlopen(PYPI_SIMPLE_API_URL) as response:
        html = response.read().decode()

    return {
        name: PYPI_SIMPLE_API_URL + url
        for url, name in re.findall(
            f'<a href="/simple/(.+)">({prefix}.*)</a>', html
        )
    }


@lru_cache(maxsize=128)
def get_packages_by_classifier(classifier: str) -> List[str]:
    """Search for packages declaring ``classifier`` on PyPI

    Yields
    ------
    name : str
        name of all packages at pypi that declare ``classifier``
    """
    packages = []
    page = 1
    pattern = re.compile('class="package-snippet__name">(.+)</span>')
    url = f"https://pypi.org/search/?c={parse.quote_plus(classifier)}&page="
    while True:
        try:
            with request.urlopen(f'{url}{page}') as response:
                html = response.read().decode()
                packages.extend(pattern.findall(html))
            page += 1
        except error.HTTPError:
            break
    return packages


@lru_cache(maxsize=128)
def get_package_versions(name: str) -> List[str]:
    """Get available versions of a package on pypi

    Parameters
    ----------
    name : str
        name of the package

    Returns
    -------
    tuple
        versions available on pypi
    """
    with request.urlopen(PYPI_SIMPLE_API_URL + name) as response:
        html = response.read()

    return re.findall(f'>{name}-(.+).tar', html.decode())


@lru_cache(maxsize=128)
def ensure_published_at_pypi(
    name: str, min_dev_status=3
) -> Optional[ProjectInfo]:
    """Return name if ``name`` is a package in PyPI with dev_status > min."""
    try:
        with request.urlopen(f'https://pypi.org/pypi/{name}/json') as resp:
            info = json.loads(resp.read().decode()).get("info")
    except error.HTTPError:
        return None
    classifiers = info.get("classifiers")
    for i in range(1, min_dev_status):
        if any(f'Development Status :: {1}' in x for x in classifiers):
            return None

    return ProjectInfo(
        name=normalized_name(info["name"]),
        version=info["version"],
        url=info["home_page"],
        summary=info["summary"],
        author=info["author"],
        license=info["license"] or "UNKNOWN",
    )


def iter_napari_plugin_info(
    skip={'napari-plugin-engine'},
) -> Generator[ProjectInfo, None, None]:
    """Return a generator that yields ProjectInfo of available napari plugins.

    By default, requires that packages are at least "Alpha" stage of
    development.  to allow lower, change the ``min_dev_status`` argument to
    ``ensure_published_at_pypi``.
    """
    already_yielded = set()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(ensure_published_at_pypi, name)
            for name in get_packages_by_classifier("Framework :: napari")
            if name not in skip
        ]

        for future in as_completed(futures):
            info = future.result()
            if info and info not in already_yielded:
                already_yielded.add(info)
                yield info


def normalized_name(name) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

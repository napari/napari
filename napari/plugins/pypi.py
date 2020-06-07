"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import configparser
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Generator
from urllib import error, request

PYPI_SIMPLE_API_URL = 'https://pypi.org/simple/'


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
            f'<a href="/simple/(.+)">({prefix}.+)</a>', html
        )
    }


@lru_cache(maxsize=128)
def get_package_versions(name: str) -> Tuple[str]:
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

    return tuple(set(re.findall(f'>{name}-(.+).tar', html.decode())))


@lru_cache(maxsize=1)
def get_napari_plugin_repos() -> List[dict]:
    """Return GitHub API hits for repos with "napari plugin" in the README."""
    with request.urlopen(
        'https://api.github.com/search/repositories?q="napari+plugin"+in:readme'
    ) as response:
        return json.loads(response.read().decode()).get("items")


@lru_cache(maxsize=128)
def ensure_published_at_pypi(name: str, min_dev_status=3) -> Optional[str]:
    """Return name if ``name`` is a package in PyPI with dev_status > min."""
    try:
        with request.urlopen(f'https://pypi.org/pypi/{name}/json') as resp:
            out = json.loads(resp.read().decode())
    except error.HTTPError:
        return None
    classifiers = out.get("info").get("classifiers")
    for i in range(1, min_dev_status):
        if any(f'Development Status :: {1}' in x for x in classifiers):
            return None
    return name


setup_py_entrypoint = re.compile(
    r"entry_points\s?=\s?([^}]*napari.plugin[^}]*)}"
)
setup_py_pypi_name = re.compile(
    r"setup\s?\(.*name\s?=\s?['\"]([^'\"]+)['\"]", re.DOTALL
)


def ensure_published_plugin(repo_info: dict) -> Optional[str]:
    """Return name of published napari plugin or None.

    ``repo_info`` is a dict from the github API, as returned by
    get_napari_plugin_repos().
    """
    # assume repos starting with napari are following naming convention
    if repo_info['name'].lower().startswith("napari"):
        if ensure_published_at_pypi(repo_info['name']):
            return repo_info['name']
        return None

    # otherwise... we have to look for the entry_point
    raw_url = 'https://raw.githubusercontent.com/{}/{}/'
    base = raw_url.format(repo_info['full_name'], repo_info['default_branch'])

    # first check setup.py
    try:
        with request.urlopen(base + "setup.py") as resp:
            text = resp.read().decode()
        match = setup_py_entrypoint.search(text)
        if match:
            if any(
                'napari.plugin' in line.split("#")[0]
                for line in match.groups()[0].splitlines()
            ):
                name = setup_py_pypi_name.search(text)
                if name and ensure_published_at_pypi(name.groups()[0]):
                    return name.groups()[0]
    except error.HTTPError:
        pass

    # then check setup.cfg
    try:
        with request.urlopen(base + "setup.cfg") as resp:
            text = resp.read().decode()
        parser = configparser.ConfigParser()
        parser.read_string(text)
        if parser.has_option('options.entry_points', 'napari.plugin'):
            return parser.get("metadata", "name")
    except error.HTTPError:
        pass
    return None


def iter_napari_plugin_names() -> Generator[str, None, None]:
    """Return a generator that yields valid napari plugin names from."""
    already_yielded = set()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures_to_name = [
            executor.submit(ensure_published_at_pypi, name)
            for name in get_packages_by_prefix("napari")
        ]
        futures_to_name.extend(
            [
                executor.submit(ensure_published_plugin, i)
                for i in get_napari_plugin_repos()
            ]
        )
        for future in as_completed(futures_to_name):
            name = future.result()
            if name and name not in already_yielded:
                already_yielded.add(name)
                yield name

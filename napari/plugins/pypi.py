"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.
"""
import requests
import re

PYPI_SIMPLE_API_URL = 'https://pypi.org/simple/'
URL_CACHE = {}  # {name: url} for packages at pypi.org/simple
VERSION_CACHE = {}  # {name: tuple of versions} for packages at pypi.org/simple


def clear_cache():
    global URL_CACHE
    global VERSION_CACHE

    URL_CACHE = {}
    VERSION_CACHE = {}


def get_packages_by_prefix(prefix) -> dict:
    """Search for packages starting with ``prefix`` on pypi.

    Packages using naming convention: http://bit.ly/pynaming-convention
    can be autodiscovered on pypi using the SIMPLE API:
    https://www.python.org/dev/peps/pep-0503/

    Returns
    -------
    dict
        {name: url} for all packages at pypi that start with ``prefix``
    """

    response = requests.get(PYPI_SIMPLE_API_URL)
    if response.status_code == 200:
        pattern = f'<a href="/simple/(.+)">({prefix}.*)</a>'
        urls = {
            name: PYPI_SIMPLE_API_URL + url
            for url, name in re.findall(pattern, response.text)
        }
        URL_CACHE.update(urls)
        return urls


def get_package_versions(name: str = None) -> tuple:
    """Get available versions of a package on pypi

    Parameters
    ----------
    name : str
        name of the package

    Returns
    -------
    tuple of versions availabe on pypi
    """
    url = URL_CACHE.get(name, PYPI_SIMPLE_API_URL + name)
    response = requests.get(url)
    response.raise_for_status()
    versions = tuple(set(re.findall(f'>{name}-(.+).tar', response.text)))
    VERSION_CACHE[name] = versions
    return versions

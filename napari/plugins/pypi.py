"""
These convenience functions will be useful for searching pypi for packages
that match the plugin naming convention, and retrieving related metadata.

Right now there are two matching mechanism: pypi classifier and github topics.
If the repo is setup correctly in setup.py as a napari plugin, it would
also show up in napari plugin installation menu when user open napari.

However, though currently napari users would be able to see github repos tagged
with the "napari-plugin" topic <https://github.com/topics/napari-plugin>,
we could deprecate github repo matching in the future to only support matching
via PyPI with 'Framework :: napari' [classifier] to unify practices of
tagging plugin packages.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Generator, List, NamedTuple, Optional, Tuple
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
    -------
    name : str
        name of all packages at pypi that declare ``classifier``
    """

    url = f"https://pypi.org/search/?c={parse.quote_plus(classifier)}"
    with request.urlopen(url) as response:
        html = response.read().decode()

    return re.findall('class="package-snippet__name">(.+)</span>', html)


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


@lru_cache(maxsize=1)
def get_napari_plugin_repos_from_github() -> List[dict]:
    """Return GitHub API hits for repos with "napari plugin" in the README."""
    with request.urlopen(
        'https://api.github.com/search/repositories?q=topic:napari-plugin'
    ) as response:
        data = json.loads(response.read().decode())
        return list(data.get("items"))


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


@lru_cache(maxsize=128)
def ensure_repo_is_napari_plugin(
    info: Tuple[str, str]
) -> Optional[ProjectInfo]:
    """Return ProjectInfo of published napari plugin or None.

    This function looks for a setup.py or setup.cfg file in the default branch
    of the repo, then looks for a "napari.plugins" in the entry_points section.
    As such, it will only currently find projects that use setuptools.

    Parameters
    ----------
    info : tuple
        2-tuple containing full repo name on github (e.g. "napari/napari"), and
        branch to query (e.g. "master")

    Returns
    -------
    info : ProjectInfo, optional
        named tuple with project info or None

    """
    # otherwise... we have to look for the entry_point
    base = 'https://raw.githubusercontent.com/{}/{}/'.format(*info)

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
                if name:
                    return ensure_published_at_pypi(name.groups()[0])
    except error.HTTPError:  # usually 404
        pass

    # then check setup.cfg
    try:
        import configparser

        with request.urlopen(base + "setup.cfg") as resp:
            text = resp.read().decode()
        parser = configparser.ConfigParser()
        parser.read_string(text)
        if parser.has_option('options.entry_points', 'napari.plugin'):
            return ensure_published_at_pypi(parser.get("metadata", "name"))
    except error.HTTPError:
        pass
    return None


def iter_napari_plugin_info(
    skip={'napari-plugin-engine'},
    search_github=True,
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

        if search_github:
            futures.extend(
                [
                    executor.submit(
                        ensure_repo_is_napari_plugin,
                        (repo_info['full_name'], repo_info['default_branch']),
                    )
                    for repo_info in get_napari_plugin_repos_from_github()
                    if repo_info['name']
                    not in skip  # repo may not have pkg name
                ]
            )
        for future in as_completed(futures):
            info = future.result()
            if info and info not in already_yielded:
                already_yielded.add(info)
                yield info


@lru_cache(maxsize=1)
def list_outdated() -> Dict[str, Tuple[str, ...]]:
    # slow!
    import subprocess

    from ..utils._appdirs import user_site_packages

    env = os.environ.copy()
    combined = os.pathsep.join(
        [user_site_packages(), env.get("PYTHONPATH", "")]
    )
    env['PYTHONPATH'] = combined
    result = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'list', '--outdated'], env=env
    )
    lines = result.decode().splitlines()
    if len(lines) <= 2:
        return {}
    out = dict()
    for line in lines[2:]:
        name, *rest = line.split()
        out[name] = tuple(rest[:2])
    return out


def normalized_name(name) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
